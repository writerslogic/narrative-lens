import { createHash } from 'node:crypto';
import { createServer } from 'node:http';
import { copyFileSync, mkdirSync, readdirSync, readFileSync, statSync, writeFileSync } from 'node:fs';
import { basename, join, resolve } from 'node:path';
import { spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { isDeepStrictEqual } from 'node:util';
import { gunzipSync } from 'node:zlib';

// Node lines the package promises to run on (engines.node >=18, active LTS, current).
export const supportedNodeMajors = [18, 20, 22, 24, 26];

// Node 18-24 are Tier 1 on glibc >= 2.28 (nodejs/node BUILDING.md), so the Linux addon must not need more.
export const glibcFloor = '2.28';
const newerVersion = (a, b) => {
  const x = a.split('.').map(Number);
  const y = b.split('.').map(Number);
  for (let index = 0; index < 3; index += 1) if ((x[index] ?? 0) !== (y[index] ?? 0)) return (x[index] ?? 0) > (y[index] ?? 0);
  return false;
};

export const platforms = {
  'x86_64-apple-darwin': { suffix: 'darwin-x64', os: 'darwin', cpu: 'x64', format: 'macho', machine: 0x01000007 },
  'aarch64-apple-darwin': { suffix: 'darwin-arm64', os: 'darwin', cpu: 'arm64', format: 'macho', machine: 0x0100000c },
  'x86_64-unknown-linux-gnu': { suffix: 'linux-x64-gnu', os: 'linux', cpu: 'x64', libc: 'glibc', format: 'elf', machine: 0x3e },
  'x86_64-pc-windows-msvc': { suffix: 'win32-x64-msvc', os: 'win32', cpu: 'x64', format: 'pe', machine: 0x8664 },
};

const manifestName = 'release-manifest.json';
const maxTarballBytes = 512 * 1024 * 1024;

const requiredNotices = pkg => ['LICENSE', ...pkg.files?.includes('THIRD_PARTY_NOTICES') ? ['THIRD_PARTY_NOTICES'] : []];

export function platform(target) {
  const value = platforms[target];
  if (!value) throw new Error(`unsupported release target: ${target}`);
  return value;
}

export function digest(path) {
  return createHash('sha256').update(readFileSync(path)).digest('hex');
}

const sha256 = buffer => createHash('sha256').update(buffer).digest('hex');
const readJson = path => JSON.parse(readFileSync(path, 'utf8'));
// Git's Windows checkout can change line endings without changing the payload.
const textDigest = buffer => sha256(Buffer.from(buffer.toString('utf8').replace(/\r\n/g, '\n')));
function ensure(condition, message) {
  if (!condition) throw new Error(message);
}

export function packagePayload(root) {
  const pkg = readJson(join(root, 'package.json'));
  return Object.fromEntries([...pkg.files, 'README.md', 'LICENSE'].map(filename => [filename, textDigest(readFileSync(join(root, filename)))]));
}

/** Validates the native module container, CPU and N-API registration for a target. */
export function inspectNative(buffer, target) {
  const host = platform(target);
  const reject = reason => { throw new Error(`invalid native artifact for ${target}: ${reason}`); };
  let format;
  let machine;
  if (buffer.length >= 8 && buffer.readUInt32LE(0) === 0xfeedfacf) {
    format = 'macho';
    machine = buffer.readUInt32LE(4);
  } else if (buffer.length >= 20 && buffer.readUInt32BE(0) === 0x7f454c46) {
    format = 'elf';
    if (buffer[4] !== 2 || buffer[5] !== 1) reject('not a 64-bit little-endian ELF object');
    if (buffer.readUInt16LE(16) !== 3) reject('ELF object is not a shared library');
    machine = buffer.readUInt16LE(18);
  } else if (buffer.length >= 64 && buffer.readUInt16LE(0) === 0x5a4d) {
    const header = buffer.readUInt32LE(0x3c);
    if (header + 6 > buffer.length || buffer.readUInt32LE(header) !== 0x00004550) reject('truncated PE header');
    format = 'pe';
    machine = buffer.readUInt16LE(header + 4);
  } else {
    reject('unrecognized binary format');
  }
  if (format !== host.format || machine !== host.machine) reject(`${format} machine 0x${machine.toString(16)} does not match ${host.format} 0x${host.machine.toString(16)}`);
  if (!buffer.includes('napi_register_module_v1')) reject('missing N-API module registration');
  if (format !== 'elf') return {};
  const versions = [...buffer.toString('latin1').matchAll(/GLIBC_(\d+)\.(\d+)(?:\.(\d+))?/g)].map(match => match.slice(1).map(part => Number(part ?? 0)));
  versions.sort((a, b) => a[0] - b[0] || a[1] - b[1] || (a[2] ?? 0) - (b[2] ?? 0));
  const highest = versions.at(-1);
  return highest ? { glibcMax: highest.slice(0, highest[2] ? 3 : 2).join('.') } : {};
}

function field(header, start, length) {
  const end = header.indexOf(0, start);
  return header.toString('utf8', start, end >= 0 && end < start + length ? end : start + length);
}

/** Reads an npm-style gzip ustar archive into a map of package-relative paths. */
export function readTarball(path) {
  const data = gunzipSync(readFileSync(path), { maxOutputLength: maxTarballBytes });
  const entries = new Map();
  let offset = 0;
  let override;
  while (offset + 512 <= data.length) {
    const header = data.subarray(offset, offset + 512);
    if (header.every(byte => byte === 0)) break;
    let sum = 0;
    for (let index = 0; index < 512; index += 1) sum += index >= 148 && index < 156 ? 32 : header[index];
    ensure(sum === Number.parseInt(field(header, 148, 8).trim(), 8), `corrupt tar header in ${basename(path)}`);
    const size = Number.parseInt(field(header, 124, 12).trim() || '0', 8);
    const type = String.fromCharCode(header[156] || 48);
    offset += 512;
    ensure(Number.isSafeInteger(size) && size >= 0 && offset + size <= data.length, `truncated tar entry in ${basename(path)}`);
    const body = data.subarray(offset, offset + size);
    offset += Math.ceil(size / 512) * 512;
    if (type === 'g') continue;
    if (type === 'x') {
      const record = /\d+ path=([^\n]*)\n/.exec(body.toString('utf8'));
      override = record?.[1];
      continue;
    }
    const prefix = field(header, 345, 155);
    const name = override ?? (prefix ? `${prefix}/${field(header, 0, 100)}` : field(header, 0, 100));
    override = undefined;
    if (type === '5') continue;
    ensure(type === '0', `unsupported tar entry type ${type} in ${basename(path)}: ${name}`);
    const segments = name.split('/');
    ensure(segments[0] === 'package' && segments.length >= 2 && segments.every(segment => segment && segment !== '.' && segment !== '..' && !segment.includes('\\')), `unsafe tar path in ${basename(path)}: ${name}`);
    const relative = segments.slice(1).join('/');
    ensure(!entries.has(relative), `duplicate tar entry in ${basename(path)}: ${relative}`);
    entries.set(relative, Buffer.from(body));
  }
  return entries;
}

function describePack(path) {
  const bytes = readFileSync(path);
  const files = readTarball(path);
  ensure(files.has('package.json'), `tarball has no package.json: ${basename(path)}`);
  return {
    files,
    metadata: JSON.parse(files.get('package.json').toString('utf8')),
    sha256: sha256(bytes),
    integrity: `sha512-${createHash('sha512').update(bytes).digest('base64')}`,
    shasum: createHash('sha1').update(bytes).digest('hex'),
  };
}

export function releasePlan(root) {
  const pkg = readJson(join(root, 'package.json'));
  ensure(/^[a-z0-9][a-z0-9._-]{0,127}$/.test(pkg.name), 'invalid package name');
  ensure(/^[A-Za-z0-9._-]{1,128}$/.test(pkg.napi.binaryName), 'invalid native binary name');
  const targets = pkg.napi.targets;
  ensure(Array.isArray(targets) && targets.length > 0 && new Set(targets).size === targets.length, 'release targets must be a nonempty unique list');
  return {
    pkg,
    rootFilename: `${pkg.name}-${pkg.version}.tgz`,
    packages: targets.map(target => {
      const host = platform(target);
      return { target, host, name: `${pkg.name}-${host.suffix}`, filename: `${pkg.name}-${host.suffix}-${pkg.version}.tgz`, binary: `${pkg.napi.binaryName}.${host.suffix}.node` };
    }),
  };
}

export function stageNotices(root, targets, directory = 'npm') {
  const pkg = readJson(join(root, 'package.json'));
  for (const target of targets) {
    if (!pkg.napi.targets.includes(target)) throw new Error(`unconfigured release target: ${target}`);
    const folder = join(root, directory, platform(target).suffix);
    const filename = join(folder, 'package.json');
    const metadata = readJson(filename);
    if (metadata.name !== `${pkg.name}-${platform(target).suffix}` || metadata.version !== pkg.version) throw new Error('invalid notice destination package');
    // The platform tarball carries the compiled Rust dependencies, so it needs the license text too.
    for (const notice of requiredNotices(pkg)) copyFileSync(join(root, notice), join(folder, notice));
    if (pkg.files?.includes('THIRD_PARTY_NOTICES')) metadata.files = [...new Set([...(metadata.files || []), 'THIRD_PARTY_NOTICES'])];
    writeFileSync(filename, JSON.stringify(metadata, null, 2) + '\n');
  }
}

export function verifyArtifacts(root, targets, directory = 'npm') {
  const { pkg } = releasePlan(root);
  if (!Array.isArray(targets) || targets.length === 0 || new Set(targets).size !== targets.length) {
    throw new Error('release targets must be a nonempty unique list');
  }
  return targets.map(target => {
    if (!pkg.napi.targets.includes(target)) throw new Error(`unconfigured release target: ${target}`);
    const host = platform(target);
    const folder = join(root, directory, host.suffix);
    const path = join(folder, `${pkg.napi.binaryName}.${host.suffix}.node`);
    const metadata = readJson(join(folder, 'package.json'));
    if (metadata.name !== `${pkg.name}-${host.suffix}` || metadata.version !== pkg.version
      || !metadata.os?.includes(host.os) || !metadata.cpu?.includes(host.cpu)
      || host.libc && !metadata.libc?.includes(host.libc)) {
      throw new Error(`invalid platform package metadata: ${target}`);
    }
    const stat = statSync(path);
    if (!stat.isFile() || stat.size === 0) throw new Error(`missing or empty native artifact: ${target}`);
    inspectNative(readFileSync(path), target);
    for (const notice of requiredNotices(pkg)) {
      if (digest(join(folder, notice)) !== digest(join(root, notice))
        || notice === 'THIRD_PARTY_NOTICES' && !metadata.files?.includes(notice)) {
        throw new Error(`missing or changed ${notice === 'LICENSE' ? 'license' : 'third-party notices'}: ${target}`);
      }
    }
    return { target, path, filename: basename(path), package: metadata.name, version: pkg.version, sha256: digest(path) };
  });
}

/** Derives every release fact from the tarballs themselves. */
export function describeRelease(root, directory, commit) {
  const plan = releasePlan(root);
  const rootPack = describePack(join(directory, plan.rootFilename));
  return {
    schema: 1,
    name: plan.pkg.name,
    version: plan.pkg.version,
    commit,
    root: { filename: plan.rootFilename, sha256: rootPack.sha256, integrity: rootPack.integrity },
    platforms: plan.packages.map(item => {
      const pack = describePack(join(directory, item.filename));
      const binary = pack.files.get(item.binary);
      ensure(binary?.length, `missing or empty native artifact: ${item.target}`);
      return {
        target: item.target,
        package: pack.metadata.name,
        filename: item.filename,
        sha256: pack.sha256,
        integrity: pack.integrity,
        nativeSha256: sha256(binary),
        ...inspectNative(binary, item.target),
      };
    }),
    payload: packagePayload(root),
  };
}

export function packRelease(root, directory, commit) {
  ensure(/^[0-9a-f]{40}$/.test(commit ?? ''), 'a full 40-character commit SHA is required');
  const plan = releasePlan(root);
  mkdirSync(directory, { recursive: true });
  const [command, prefix] = process.env.npm_execpath ? [process.execPath, [process.env.npm_execpath]] : ['npm', []];
  const pack = folder => {
    const result = spawnSync(command, [...prefix, 'pack', '--ignore-scripts', '--json', '--pack-destination', directory], { cwd: folder, encoding: 'utf8', shell: !process.env.npm_execpath && process.platform === 'win32' });
    ensure(!result.error && result.status === 0, `npm pack failed in ${folder}: ${(result.stderr || result.error?.message || '').slice(-4000)}`);
  };
  pack(root);
  for (const item of plan.packages) pack(join(root, 'npm', item.host.suffix));
  const manifest = describeRelease(root, directory, commit);
  writeFileSync(join(directory, manifestName), JSON.stringify(manifest, null, 2) + '\n');
  return verifyRelease(root, directory, { commit });
}

/** Rejects a missing, stale, extra or tampered assembled release; returns its manifest. */
export function verifyRelease(root, directory, { commit } = {}) {
  ensure(/^[0-9a-f]{40}$/.test(commit ?? ''), 'the expected commit SHA is required');
  const plan = releasePlan(root);
  const present = new Set(readdirSync(directory));
  ensure(present.has(manifestName), `missing release file: ${manifestName}`);
  const manifest = readJson(join(directory, manifestName));
  ensure(manifest.schema === 1, 'unsupported release manifest schema');
  ensure(manifest.name === plan.pkg.name && manifest.version === plan.pkg.version, `stale release: manifest is ${manifest.name}@${manifest.version}, expected ${plan.pkg.name}@${plan.pkg.version}`);
  ensure(manifest.commit === commit, `release was assembled from ${manifest.commit}, expected ${commit}`);
  for (const name of [plan.rootFilename, ...plan.packages.map(item => item.filename)]) ensure(present.has(name), `missing release file: ${name}`);
  const expected = new Set([manifestName, plan.rootFilename, ...plan.packages.map(item => item.filename)]);
  for (const name of readdirSync(directory)) ensure(expected.has(name), `unexpected release file: ${name}`);
  ensure(isDeepStrictEqual(manifest.platforms?.map(item => item.target), plan.packages.map(item => item.target)), 'release targets differ from the configured targets');

  // Compare raw bytes first so a modified archive is never parsed.
  ensure(digest(join(directory, plan.rootFilename)) === manifest.root?.sha256, 'root tarball digest changed after assembly');
  for (const [index, item] of plan.packages.entries()) {
    ensure(digest(join(directory, item.filename)) === manifest.platforms[index].sha256, `platform tarball digest changed after assembly: ${item.target}`);
  }
  const actual = describeRelease(root, directory, manifest.commit);
  for (const [index, item] of plan.packages.entries()) {
    const listed = manifest.platforms[index];
    const found = actual.platforms[index];
    ensure(found.nativeSha256 === listed.nativeSha256, `native binary differs from the release manifest: ${item.target}`);
    ensure(!item.host.libc || !newerVersion(found.glibcMax ?? '0', glibcFloor), `native module for ${item.target} needs glibc ${found.glibcMax}, newer than the ${glibcFloor} that Node 18-24 support`);
    const pack = describePack(join(directory, item.filename));
    const metadata = pack.metadata;
    if (metadata.name !== item.name || metadata.version !== plan.pkg.version
      || !isDeepStrictEqual(metadata.os, [item.host.os]) || !isDeepStrictEqual(metadata.cpu, [item.host.cpu])
      || !isDeepStrictEqual(metadata.libc, item.host.libc ? [item.host.libc] : undefined)
      || metadata.main !== item.binary || metadata.license !== plan.pkg.license || !isDeepStrictEqual(metadata.engines, plan.pkg.engines)) {
      throw new Error(`invalid platform package metadata: ${item.target}`);
    }
    const notices = requiredNotices(plan.pkg);
    const required = ['package.json', ...notices, item.binary];
    const allowed = new Set([...required, 'README.md']);
    for (const name of pack.files.keys()) ensure(allowed.has(name), `unexpected file in ${item.name}: ${name}`);
    for (const name of required) ensure(pack.files.has(name), `${item.name} is missing ${name}`);
    for (const notice of notices) {
      ensure(textDigest(pack.files.get(notice)) === textDigest(readFileSync(join(root, notice))), `missing or changed ${notice === 'LICENSE' ? 'license' : 'third-party notices'}: ${item.target}`);
    }
  }

  const rootPack = describePack(join(directory, plan.rootFilename));
  const { optionalDependencies, ...published } = rootPack.metadata;
  const { optionalDependencies: ignored, ...source } = plan.pkg;
  ensure(isDeepStrictEqual(published, source), 'root package.json differs from the repository package.json');
  ensure(isDeepStrictEqual(Object.entries(optionalDependencies ?? {}).sort(), plan.packages.map(item => [item.name, plan.pkg.version]).sort()), 'root optionalDependencies must pin every platform package at the root version');
  const rootFiles = new Set([...plan.pkg.files, 'package.json', 'README.md', 'LICENSE']);
  for (const name of rootPack.files.keys()) ensure(rootFiles.has(name), `unexpected file in ${plan.pkg.name}: ${name}`);
  for (const name of rootFiles) ensure(rootPack.files.has(name), `${plan.pkg.name} is missing ${name}`);
  const payload = packagePayload(root);
  for (const [name, hash] of Object.entries(payload)) ensure(textDigest(rootPack.files.get(name)) === hash, `release payload differs from the repository: ${name}`);
  ensure(isDeepStrictEqual(manifest.payload, payload), 'release payload differs from the repository');
  ensure(isDeepStrictEqual(manifest, actual), 'release manifest differs from the tarballs');
  return manifest;
}

export function verifyAttestations(root, directory, { commit, manifestSha256, manifest }) {
  const plan = releasePlan(root);
  const files = [];
  const visit = folder => {
    for (const entry of readdirSync(folder, { withFileTypes: true })) {
      const path = join(folder, entry.name);
      if (entry.isDirectory()) visit(path);
      else if (entry.name.endsWith('.json')) files.push(path);
    }
  };
  visit(directory);
  const seen = new Map();
  const typed = new Set();
  for (const file of files) {
    const item = readJson(file);
    const label = `${basename(file)}`;
    const entry = plan.packages.find(candidate => candidate.target === item.target);
    ensure(item.schema === 1 && item.kind === 'smoke' && entry, `invalid smoke attestation: ${label}`);
    ensure(item.commit === commit && item.version === plan.pkg.version, `stale smoke attestation: ${label}`);
    ensure(item.releaseManifestSha256 === manifestSha256, `smoke attestation is for a different release: ${label}`);
    ensure(item.platform === entry.host.os && item.arch === entry.host.cpu && (!entry.host.libc || item.libc === entry.host.libc), `smoke attestation ran on the wrong platform: ${label}`);
    const major = Number.parseInt(item.node, 10);
    ensure(supportedNodeMajors.includes(major) && /^\d+\.\d+\.\d+/.test(item.node), `smoke attestation has an unsupported Node version: ${label}`);
    const built = manifest.platforms.find(candidate => candidate.target === item.target);
    ensure(item.nativeSha256 === built.nativeSha256 && isDeepStrictEqual(item.archives, [{ filename: manifest.root.filename, sha256: manifest.root.sha256 }, { filename: built.filename, sha256: built.sha256 }]), `smoke attestation tested different archives: ${label}`);
    ensure(item.registryResolution === true && Array.isArray(item.checks) && ['cjs', 'esm', 'manuscript', 'notices', 'version-lock'].every(name => item.checks.includes(name)), `smoke attestation is incomplete: ${label}`);
    const key = `${item.target}|${major}`;
    ensure(!seen.has(key), `duplicate smoke attestation: ${item.target} Node ${major}`);
    seen.set(key, item);
    if (item.types === true) typed.add(item.target);
  }
  for (const item of plan.packages) {
    for (const major of supportedNodeMajors) ensure(seen.has(`${item.target}|${major}`), `missing smoke attestation: ${item.target} Node ${major}`);
    ensure(typed.has(item.target), `missing TypeScript consumer attestation: ${item.target}`);
  }
  return seen.size;
}

/** Publication order: every platform package before the root that depends on them. */
export function publishOrder(manifest) {
  return [...manifest.platforms.map(item => ({ name: item.package, ...item })), { name: manifest.name, ...manifest.root }];
}

export function publishRelease(root, directory, { commit, dryRun = false, provenance = false, registry } = {}) {
  const manifest = verifyRelease(root, directory, { commit });
  const [command, prefix] = process.env.npm_execpath ? [process.execPath, [process.env.npm_execpath]] : ['npm', []];
  const npm = args => spawnSync(command, [...prefix, ...args, ...registry ? ['--registry', registry] : []], { encoding: 'utf8' });
  const tag = manifest.version.includes('-') ? ['--tag', 'next'] : [];
  const published = [];
  for (const item of publishOrder(manifest)) {
    const path = join(directory, item.filename);
    // Re-hash immediately before upload so the bytes published are the bytes verified.
    ensure(digest(path) === item.sha256, `tarball changed before publication: ${item.filename}`);
    const existing = npm(['view', `${item.name}@${manifest.version}`, 'dist.integrity']);
    if (existing.status === 0 && existing.stdout.trim()) {
      ensure(existing.stdout.trim() === item.integrity, `${item.name}@${manifest.version} is already published with different content`);
      published.push({ name: item.name, skipped: true });
      continue;
    }
    const result = npm(['publish', path, '--ignore-scripts', '--access', 'public', ...tag, ...provenance ? ['--provenance'] : [], ...dryRun ? ['--dry-run'] : []]);
    ensure(!result.error && result.status === 0, `npm publish failed for ${item.name}: ${(result.stderr || result.error?.message || '').slice(-4000)}`);
    published.push({ name: item.name, skipped: false, output: result.stderr + result.stdout });
  }
  return published;
}

/** Minimal read-only npm registry so installs exercise real optionalDependencies resolution. */
export function startRegistry(directory) {
  const packs = new Map();
  for (const name of readdirSync(directory).filter(file => file.endsWith('.tgz'))) {
    const pack = describePack(join(directory, name));
    packs.set(pack.metadata.name, { ...pack, filename: name, path: join(directory, name) });
  }
  const server = createServer((request, response) => {
    const path = decodeURIComponent(new URL(request.url, 'http://localhost').pathname.slice(1));
    const base = `http://127.0.0.1:${server.address().port}`;
    const tarball = [...packs.values()].find(pack => path === `-/${pack.filename}`);
    if (tarball) {
      response.writeHead(200, { 'content-type': 'application/octet-stream' }).end(readFileSync(tarball.path));
      return;
    }
    const pack = packs.get(path);
    if (!pack) {
      response.writeHead(404, { 'content-type': 'application/json' }).end('{"error":"not found"}');
      return;
    }
    const version = pack.metadata.version;
    response.writeHead(200, { 'content-type': 'application/json' }).end(JSON.stringify({
      name: path,
      'dist-tags': { latest: version },
      versions: { [version]: { ...pack.metadata, _id: `${path}@${version}`, dist: { tarball: `${base}/-/${pack.filename}`, shasum: pack.shasum, integrity: pack.integrity } } },
    }));
  });
  return new Promise((resolveStart, reject) => {
    server.once('error', reject);
    server.listen(0, '127.0.0.1', () => resolveStart({ url: `http://127.0.0.1:${server.address().port}`, close: () => new Promise(done => server.close(done)) }));
  });
}

function option(args, name) {
  const index = args.indexOf(name);
  if (index < 0) return undefined;
  ensure(args[index + 1] && !args[index + 1].startsWith('--'), `${name} requires a value`);
  return args[index + 1];
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const root = resolve(fileURLToPath(new URL('..', import.meta.url)));
  const args = process.argv.slice(2);
  try {
    const { pkg } = releasePlan(root);
    const commit = option(args, '--commit') ?? process.env.GITHUB_SHA;
    if (args[0] === 'pack') {
      const manifest = packRelease(root, resolve(args[1] ?? 'release-packages'), commit);
      process.stdout.write(`Packed ${manifest.platforms.length + 1} release tarballs for ${manifest.commit}\n`);
      for (const item of manifest.platforms) process.stdout.write(`  ${item.target} native ${item.nativeSha256.slice(0, 12)}${item.glibcMax ? ` needs glibc >= ${item.glibcMax}` : ''}\n`);
    } else if (args[0] === 'publish') {
      const results = publishRelease(root, resolve(args[1] ?? 'release-packages'), { commit, dryRun: args.includes('--dry-run'), provenance: args.includes('--provenance'), registry: option(args, '--registry') });
      for (const item of results) process.stdout.write(`${item.skipped ? 'Already published' : args.includes('--dry-run') ? 'Dry-run verified' : 'Published'} ${item.name}\n`);
    } else {
      if (args.includes('--stage-notices')) stageNotices(root, pkg.napi.targets);
      const artifacts = verifyArtifacts(root, pkg.napi.targets);
      process.stdout.write(`Verified ${artifacts.length} release artifacts\n`);
    }
  } catch (error) {
    process.stderr.write(`Release artifact validation failed: ${error.message}\n`);
    process.exitCode = 1;
  }
}
