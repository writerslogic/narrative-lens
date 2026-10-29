const test = require('node:test');
const assert = require('node:assert/strict');
const crypto = require('node:crypto');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const zlib = require('node:zlib');

const COMMIT = 'a'.repeat(40);
const TARGETS = {
  'x86_64-apple-darwin': { suffix: 'darwin-x64', os: 'darwin', cpu: 'x64' },
  'aarch64-apple-darwin': { suffix: 'darwin-arm64', os: 'darwin', cpu: 'arm64' },
  'x86_64-unknown-linux-gnu': { suffix: 'linux-x64-gnu', os: 'linux', cpu: 'x64', libc: 'glibc' },
  'x86_64-pc-windows-msvc': { suffix: 'win32-x64-msvc', os: 'win32', cpu: 'x64' },
};
const NODES = [18, 20, 22, 24, 26];
const sha = value => crypto.createHash('sha256').update(value).digest('hex');
const scripts = () => import('../../scripts/release_artifacts.mjs');

function native(target, marker = '', { registration = true } = {}) {
  const b = Buffer.alloc(4096);
  if (target.includes('apple')) {
    b.writeUInt32LE(0xfeedfacf, 0);
    b.writeUInt32LE(target.startsWith('x86') ? 0x01000007 : 0x0100000c, 4);
  } else if (target.includes('linux')) {
    b.writeUInt32BE(0x7f454c46, 0); b[4] = 2; b[5] = 1; b.writeUInt16LE(3, 16); b.writeUInt16LE(0x3e, 18);
    b.write('GLIBC_2.17\0GLIBC_2.28\0', 2048);
  } else {
    b.writeUInt16LE(0x5a4d, 0); b.writeUInt32LE(0x80, 0x3c); b.writeUInt32LE(0x4550, 0x80); b.writeUInt16LE(0x8664, 0x84);
  }
  if (registration) b.write('napi_register_module_v1', 1024);
  b.write(marker, 3000);
  return b;
}

function tar(files) {
  const blocks = [];
  for (const [name, content] of Object.entries(files)) {
    if (name.endsWith(':type')) continue;
    const body = Buffer.from(content);
    const header = Buffer.alloc(512);
    header.write(name.startsWith('/') || name.startsWith('..') ? name : `package/${name}`, 0, 100);
    header.write('0000644\0', 100); header.write('0000000\0', 108); header.write('0000000\0', 116);
    header.write(`${body.length.toString(8).padStart(11, '0')}\0`, 124); header.write('00000000000\0', 136);
    header.write('        ', 148); header[156] = (files[`${name}:type`] ?? '0').charCodeAt(0);
    header.write('ustar\0', 257); header.write('00', 263);
    header.write(`${[...header].reduce((sum, byte) => sum + byte, 0).toString(8).padStart(6, '0')}\0 `, 148);
    blocks.push(header, body, Buffer.alloc((512 - body.length % 512) % 512));
  }
  return zlib.gzipSync(Buffer.concat([...blocks, Buffer.alloc(1024)]));
}

/** A repository root and a matching assembled release, optionally for a different version. */
async function fixture({ version = '0.1.0', targets = Object.keys(TARGETS) } = {}) {
  const lib = await scripts();
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'narrative-lens-release-'));
  const release = path.join(root, 'release-packages');
  fs.mkdirSync(release);
  const pkg = { name: 'narrative-lens', version: '0.1.0', license: 'Apache-2.0', main: 'index.js', files: ['index.js', 'THIRD_PARTY_NOTICES'], engines: { node: '>=18' }, napi: { binaryName: 'narrative-lens', targets } };
  const texts = { 'index.js': 'module.exports = {};\n', 'README.md': 'readme\n', LICENSE: 'Apache license text\n', THIRD_PARTY_NOTICES: 'Resource copyright and license\n' };
  fs.writeFileSync(path.join(root, 'package.json'), JSON.stringify(pkg));
  for (const [name, text] of Object.entries(texts)) fs.writeFileSync(path.join(root, name), text);
  const optionalDependencies = Object.fromEntries(targets.map(target => [`narrative-lens-${TARGETS[target].suffix}`, version]));
  fs.writeFileSync(path.join(release, `narrative-lens-${version}.tgz`), tar({ ...texts, 'package.json': JSON.stringify({ ...pkg, version, optionalDependencies }) }));
  for (const target of targets) {
    const host = TARGETS[target];
    const metadata = { name: `narrative-lens-${host.suffix}`, version, os: [host.os], cpu: [host.cpu], ...host.libc && { libc: [host.libc] }, main: `narrative-lens.${host.suffix}.node`, files: [`narrative-lens.${host.suffix}.node`, 'THIRD_PARTY_NOTICES'], license: 'Apache-2.0', engines: { node: '>=18' } };
    fs.writeFileSync(path.join(release, `narrative-lens-${host.suffix}-${version}.tgz`), tar({ 'package.json': JSON.stringify(metadata), 'README.md': 'stub\n', LICENSE: texts.LICENSE, THIRD_PARTY_NOTICES: texts.THIRD_PARTY_NOTICES, [`narrative-lens.${host.suffix}.node`]: native(target, 'original') }));
  }
  const context = {
    lib, root, release, pkg, texts,
    // Re-describe the tarballs, as an attacker or a stale job would leave a self-consistent manifest.
    refresh(commit = COMMIT) {
      const manifest = lib.describeRelease(root, release, commit);
      fs.writeFileSync(path.join(release, 'release-manifest.json'), JSON.stringify(manifest, null, 2));
      return manifest;
    },
    rewrite(filename, change) {
      const file = path.join(release, filename);
      const entries = Object.fromEntries([...lib.readTarball(file)].map(([name, body]) => [name, body]));
      change(entries);
      fs.writeFileSync(file, tar(entries));
    },
    cleanup: () => fs.rmSync(root, { recursive: true, force: true }),
  };
  if (version === '0.1.0') context.refresh();
  else fs.writeFileSync(path.join(release, 'release-manifest.json'), JSON.stringify({ schema: 1, name: 'narrative-lens', version, commit: COMMIT }));
  return context;
}

test('native modules must match the target container, CPU and N-API registration', async () => {
  const { inspectNative } = await scripts();
  for (const target of Object.keys(TARGETS)) inspectNative(native(target), target);
  assert.throws(() => inspectNative(native('x86_64-apple-darwin'), 'aarch64-apple-darwin'), /does not match/);
  assert.throws(() => inspectNative(native('x86_64-unknown-linux-gnu'), 'x86_64-apple-darwin'), /does not match/);
  assert.throws(() => inspectNative(native('x86_64-pc-windows-msvc'), 'x86_64-unknown-linux-gnu'), /does not match/);
  assert.throws(() => inspectNative(Buffer.from('tested binary'), 'aarch64-apple-darwin'), /unrecognized binary format/);
  assert.throws(() => inspectNative(native('aarch64-apple-darwin').subarray(0, 6), 'aarch64-apple-darwin'), /unrecognized binary format/);
  assert.throws(() => inspectNative(native('aarch64-apple-darwin', '', { registration: false }), 'aarch64-apple-darwin'), /N-API/);
  assert.equal(inspectNative(native('x86_64-unknown-linux-gnu'), 'x86_64-unknown-linux-gnu').glibcMax, '2.28');
});

test('release preflight rejects missing targets, empty binaries and wrong package identity', async () => {
  const { verifyArtifacts } = await scripts();
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'narrative-lens-preflight-'));
  const target = 'aarch64-apple-darwin';
  try {
    fs.writeFileSync(path.join(root, 'package.json'), JSON.stringify({ name: 'narrative-lens', version: '0.1.0', napi: { binaryName: 'narrative-lens', targets: [target] } }));
    fs.writeFileSync(path.join(root, 'LICENSE'), 'license\n');
    assert.throws(() => verifyArtifacts(root, [target]), /ENOENT/);
    const directory = path.join(root, 'npm', 'darwin-arm64');
    fs.mkdirSync(directory, { recursive: true });
    const metadata = { name: 'narrative-lens-darwin-arm64', version: '0.1.0', os: ['darwin'], cpu: ['arm64'] };
    const manifest = path.join(directory, 'package.json');
    fs.writeFileSync(manifest, JSON.stringify(metadata));
    fs.writeFileSync(path.join(directory, 'narrative-lens.darwin-arm64.node'), '');
    assert.throws(() => verifyArtifacts(root, [target]), /empty native artifact/);
    fs.writeFileSync(path.join(directory, 'narrative-lens.darwin-arm64.node'), native('x86_64-apple-darwin'));
    assert.throws(() => verifyArtifacts(root, [target]), /does not match/);
    fs.writeFileSync(path.join(directory, 'narrative-lens.darwin-arm64.node'), native(target));
    assert.throws(() => verifyArtifacts(root, [target]), /ENOENT|license/);
    fs.copyFileSync(path.join(root, 'LICENSE'), path.join(directory, 'LICENSE'));
    assert.equal(verifyArtifacts(root, [target]).length, 1);
    metadata.version = '0.0.0';
    fs.writeFileSync(manifest, JSON.stringify(metadata));
    assert.throws(() => verifyArtifacts(root, [target]), /metadata/);
    assert.throws(() => verifyArtifacts(root, [target, target]), /unique/);
    assert.throws(() => verifyArtifacts(root, ['../../unconfigured']), /unconfigured/);
  } finally { fs.rmSync(root, { recursive: true, force: true }); }
});

test('platform packages must bundle the license and unchanged third-party notices', async () => {
  const { stageNotices, verifyArtifacts } = await scripts();
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'narrative-lens-notices-'));
  const target = 'aarch64-apple-darwin';
  try {
    fs.writeFileSync(path.join(root, 'package.json'), JSON.stringify({ name: 'narrative-lens', version: '0.1.0', files: ['THIRD_PARTY_NOTICES'], napi: { binaryName: 'narrative-lens', targets: [target] } }));
    fs.writeFileSync(path.join(root, 'THIRD_PARTY_NOTICES'), 'Resource copyright and license\n');
    fs.writeFileSync(path.join(root, 'LICENSE'), 'Apache license text\n');
    const platform = path.join(root, 'npm', 'darwin-arm64');
    fs.mkdirSync(platform, { recursive: true });
    fs.writeFileSync(path.join(platform, 'package.json'), JSON.stringify({ name: 'narrative-lens-darwin-arm64', version: '0.1.0', os: ['darwin'], cpu: ['arm64'], files: ['narrative-lens.darwin-arm64.node'] }));
    fs.writeFileSync(path.join(platform, 'narrative-lens.darwin-arm64.node'), native(target));
    assert.throws(() => verifyArtifacts(root, [target]), /ENOENT|notices|license/);
    stageNotices(root, [target]);
    assert.equal(verifyArtifacts(root, [target]).length, 1);
    for (const [file, pattern] of [['THIRD_PARTY_NOTICES', /notices/], ['LICENSE', /license/]]) {
      const original = fs.readFileSync(path.join(platform, file));
      fs.writeFileSync(path.join(platform, file), 'changed');
      assert.throws(() => verifyArtifacts(root, [target]), pattern);
      fs.writeFileSync(path.join(platform, file), original);
    }
    fs.rmSync(path.join(platform, 'LICENSE'));
    assert.throws(() => verifyArtifacts(root, [target]), /ENOENT|license/);
  } finally { fs.rmSync(root, { recursive: true, force: true }); }
});

test('tar reader rejects traversal, links and corrupt archives', async () => {
  const { readTarball } = await scripts();
  const root = fs.mkdtempSync(path.join(os.tmpdir(), 'narrative-lens-tar-'));
  try {
    const read = files => { const file = path.join(root, 'x.tgz'); fs.writeFileSync(file, tar(files)); return readTarball(file); };
    assert.equal(read({ 'a/b.txt': 'ok' }).get('a/b.txt').toString(), 'ok');
    assert.throws(() => read({ '../escape': 'x' }), /unsafe tar path/);
    assert.throws(() => read({ '/abs': 'x' }), /unsafe tar path/);
    assert.throws(() => read({ 'a/../../b': 'x' }), /unsafe tar path/);
    assert.throws(() => read({ link: '', 'link:type': '2' }), /unsupported tar entry/);
    assert.throws(() => read({ a: '1', 'b/../a': '2' }), /unsafe tar path/);
    const damaged = zlib.gunzipSync(tar({ a: 'content' }));
    damaged[0] ^= 1;
    fs.writeFileSync(path.join(root, 'bad.tgz'), zlib.gzipSync(damaged));
    assert.throws(() => readTarball(path.join(root, 'bad.tgz')), /corrupt tar header/);
  } finally { fs.rmSync(root, { recursive: true, force: true }); }
});

test('assembled release is accepted only when complete, current and untampered', async () => {
  const f = await fixture();
  try {
    const { verifyRelease } = f.lib;
    assert.equal(verifyRelease(f.root, f.release, { commit: COMMIT }).platforms.length, 4);
    assert.throws(() => verifyRelease(f.root, f.release, { commit: 'b'.repeat(40) }), /assembled from/);
    assert.throws(() => verifyRelease(f.root, f.release, {}), /expected commit/);

    const tarball = 'narrative-lens-darwin-arm64-0.1.0.tgz';
    const original = fs.readFileSync(path.join(f.release, tarball));
    fs.appendFileSync(path.join(f.release, tarball), 'x');
    assert.throws(() => verifyRelease(f.root, f.release, { commit: COMMIT }), /digest changed/);
    fs.writeFileSync(path.join(f.release, tarball), original);
    const rootTarball = path.join(f.release, 'narrative-lens-0.1.0.tgz');
    const rootOriginal = fs.readFileSync(rootTarball);
    fs.appendFileSync(rootTarball, 'x');
    assert.throws(() => verifyRelease(f.root, f.release, { commit: COMMIT }), /root tarball digest changed/);
    fs.writeFileSync(rootTarball, rootOriginal);

    // A swapped binary with a refreshed tarball digest but stale nativeSha256.
    const manifestFile = path.join(f.release, 'release-manifest.json');
    const manifest = JSON.parse(fs.readFileSync(manifestFile, 'utf8'));
    f.rewrite(tarball, files => { files['narrative-lens.darwin-arm64.node'] = native('aarch64-apple-darwin', 'swapped'); });
    manifest.platforms.find(item => item.target === 'aarch64-apple-darwin').sha256 = sha(fs.readFileSync(path.join(f.release, tarball)));
    fs.writeFileSync(manifestFile, JSON.stringify(manifest));
    assert.throws(() => verifyRelease(f.root, f.release, { commit: COMMIT }), /native binary differs/);

    // The remaining defects are self-consistent: the manifest is regenerated after each change.
    const cases = [
      ['wrong CPU in the module', t => f.rewrite(t, files => { files['narrative-lens.darwin-arm64.node'] = native('x86_64-apple-darwin'); }), tarball, /does not match/],
      ['platform tarball without LICENSE', t => f.rewrite(t, files => { delete files.LICENSE; }), tarball, /missing LICENSE|is missing/],
      ['platform tarball with stale notices', t => f.rewrite(t, files => { files.THIRD_PARTY_NOTICES = 'old'; }), tarball, /notices/],
      ['platform tarball with a stray file', t => f.rewrite(t, files => { files['.env'] = 'secret'; }), tarball, /unexpected file/],
      ['platform package for the wrong CPU', t => f.rewrite(t, files => { files['package.json'] = JSON.stringify({ ...JSON.parse(files['package.json']), cpu: ['arm64'] }); }), 'narrative-lens-linux-x64-gnu-0.1.0.tgz', /metadata/],
      ['glibc package without libc', t => f.rewrite(t, files => { const { libc, ...rest } = JSON.parse(files['package.json']); files['package.json'] = JSON.stringify(rest); }), 'narrative-lens-linux-x64-gnu-0.1.0.tgz', /metadata/],
      ['linux module needing a newer glibc than Node supports', t => f.rewrite(t, files => { files['narrative-lens.linux-x64-gnu.node'] = Buffer.from(native('x86_64-unknown-linux-gnu').toString('latin1').replace('GLIBC_2.28', 'GLIBC_2.34'), 'latin1'); }), 'narrative-lens-linux-x64-gnu-0.1.0.tgz', /needs glibc 2\.34/],
      ['root tarball carrying a binary', t => f.rewrite(t, files => { files['narrative-lens.darwin-arm64.node'] = native('aarch64-apple-darwin'); }), 'narrative-lens-0.1.0.tgz', /unexpected file/],
      ['root missing a platform dependency', t => f.rewrite(t, files => { const p = JSON.parse(files['package.json']); delete p.optionalDependencies['narrative-lens-win32-x64-msvc']; files['package.json'] = JSON.stringify(p); }), 'narrative-lens-0.1.0.tgz', /optionalDependencies/],
      ['root pinning a stale platform version', t => f.rewrite(t, files => { const p = JSON.parse(files['package.json']); p.optionalDependencies['narrative-lens-darwin-x64'] = '0.0.9'; files['package.json'] = JSON.stringify(p); }), 'narrative-lens-0.1.0.tgz', /optionalDependencies/],
      ['root metadata changed', t => f.rewrite(t, files => { files['package.json'] = JSON.stringify({ ...JSON.parse(files['package.json']), main: 'evil.js' }); }), 'narrative-lens-0.1.0.tgz', /differs from the repository package.json/],
      ['root code changed after testing', t => f.rewrite(t, files => { files['index.js'] = 'module.exports = { evil: true };\n'; }), 'narrative-lens-0.1.0.tgz', /payload differs/],
    ];
    const pristine = await fixture();
    try {
      for (const [label, mutate, file, expected] of cases) {
        for (const name of fs.readdirSync(pristine.release)) fs.copyFileSync(path.join(pristine.release, name), path.join(f.release, name));
        assert.throws(() => {
          mutate(file);
          f.refresh();
          verifyRelease(f.root, f.release, { commit: COMMIT });
        }, expected, label);
      }
      for (const name of fs.readdirSync(pristine.release)) fs.copyFileSync(path.join(pristine.release, name), path.join(f.release, name));
      assert.equal(verifyRelease(f.root, f.release, { commit: COMMIT }).platforms.length, 4);
    } finally { pristine.cleanup(); }

    fs.writeFileSync(path.join(f.release, 'stale-0.0.9.tgz'), 'old');
    assert.throws(() => verifyRelease(f.root, f.release, { commit: COMMIT }), /unexpected release file/);
    fs.rmSync(path.join(f.release, 'stale-0.0.9.tgz'));
    fs.rmSync(path.join(f.release, 'narrative-lens-win32-x64-msvc-0.1.0.tgz'));
    assert.throws(() => verifyRelease(f.root, f.release, { commit: COMMIT }), /missing release file: narrative-lens-win32-x64-msvc-0.1.0.tgz/);
  } finally { f.cleanup(); }
});

test('release assembled for another version or target set is rejected', async () => {
  const stale = await fixture({ version: '0.0.9' });
  try {
    assert.throws(() => stale.lib.verifyRelease(stale.root, stale.release, { commit: COMMIT }), /stale release/);
  } finally { stale.cleanup(); }
  const partial = await fixture({ targets: ['aarch64-apple-darwin'] });
  try {
    fs.writeFileSync(path.join(partial.root, 'package.json'), JSON.stringify({ ...partial.pkg, napi: { binaryName: 'narrative-lens', targets: Object.keys(TARGETS) } }));
    assert.throws(() => partial.lib.verifyRelease(partial.root, partial.release, { commit: COMMIT }), /targets differ|missing release file/);
  } finally { partial.cleanup(); }
});

test('payload hashes ignore checkout line endings but not content', async () => {
  const f = await fixture();
  try {
    fs.writeFileSync(path.join(f.root, 'index.js'), 'module.exports = {};\r\n');
    assert.equal(f.lib.verifyRelease(f.root, f.release, { commit: COMMIT }).platforms.length, 4);
    fs.writeFileSync(path.join(f.root, 'index.js'), 'module.exports = { changed: true };\n');
    assert.throws(() => f.lib.verifyRelease(f.root, f.release, { commit: COMMIT }), /payload differs/);
  } finally { f.cleanup(); }
});

function attestation(f, manifest, target, node, overrides = {}) {
  const host = TARGETS[target];
  const built = manifest.platforms.find(item => item.target === target);
  return {
    schema: 1, kind: 'smoke', commit: COMMIT, version: '0.1.0', target, platform: host.os, arch: host.cpu, ...host.libc && { libc: host.libc },
    node: `${node}.1.0`, npm: '10.0.0', types: node === 24, registryResolution: true, checks: ['cjs', 'esm', 'manuscript', 'notices', 'version-lock'],
    releaseManifestSha256: sha(fs.readFileSync(path.join(f.release, 'release-manifest.json'))), nativeSha256: built.nativeSha256,
    archives: [{ filename: manifest.root.filename, sha256: manifest.root.sha256 }, { filename: built.filename, sha256: built.sha256 }], ...overrides,
  };
}

function attestations(f, manifest, change = () => {}) {
  const directory = fs.mkdtempSync(path.join(f.root, 'attestations-'));
  for (const target of Object.keys(TARGETS)) {
    for (const node of NODES) {
      const folder = path.join(directory, `attestation-${target}-node${node}`);
      fs.mkdirSync(folder);
      const item = attestation(f, manifest, target, node);
      change(item, target, node);
      fs.writeFileSync(path.join(folder, 'attestation.json'), JSON.stringify(item));
    }
  }
  return directory;
}

test('smoke attestations must cover every target and supported Node line for this release', async () => {
  const f = await fixture();
  try {
    const manifest = f.lib.verifyRelease(f.root, f.release, { commit: COMMIT });
    const options = { commit: COMMIT, manifest, manifestSha256: sha(fs.readFileSync(path.join(f.release, 'release-manifest.json'))) };
    assert.deepEqual(f.lib.supportedNodeMajors, NODES);
    assert.equal(f.lib.verifyAttestations(f.root, attestations(f, manifest), options), 20);
    const reject = (change, pattern) => assert.throws(() => f.lib.verifyAttestations(f.root, attestations(f, manifest, change), options), pattern);
    reject((item, target, node) => { if (target === 'x86_64-pc-windows-msvc' && node === 18) item.target = 'aarch64-apple-darwin'; }, /wrong platform|duplicate|missing/);
    reject((item, target, node) => { if (node === 22) item.node = '23.0.0'; }, /unsupported Node|missing smoke attestation/);
    reject((item, target, node) => { if (target === 'x86_64-apple-darwin') item.types = false; }, /TypeScript consumer/);
    reject(item => { item.commit = 'b'.repeat(40); }, /stale smoke attestation/);
    reject((item, target) => { if (target === 'aarch64-apple-darwin') item.releaseManifestSha256 = '0'.repeat(64); }, /different release/);
    reject((item, target, node) => { if (target === 'x86_64-unknown-linux-gnu' && node === 20) item.libc = 'musl'; }, /wrong platform/);
    reject((item, target, node) => { if (target === 'x86_64-unknown-linux-gnu' && node === 20) item.platform = 'darwin'; }, /wrong platform/);
    reject((item, target, node) => { if (node === 26) item.archives[1].sha256 = '1'.repeat(64); }, /different archives/);
    reject((item, target, node) => { if (node === 26) item.nativeSha256 = '1'.repeat(64); }, /different archives/);
    reject((item, target, node) => { if (node === 26) item.registryResolution = false; }, /incomplete/);
    reject((item, target, node) => { if (node === 26) item.checks = ['cjs']; }, /incomplete/);
    const directory = attestations(f, manifest);
    fs.rmSync(path.join(directory, 'attestation-x86_64-apple-darwin-node20'), { recursive: true });
    assert.throws(() => f.lib.verifyAttestations(f.root, directory, options), /missing smoke attestation: x86_64-apple-darwin Node 20/);
    const duplicate = attestations(f, manifest);
    fs.mkdirSync(path.join(duplicate, 'again'));
    fs.copyFileSync(path.join(duplicate, 'attestation-x86_64-apple-darwin-node20', 'attestation.json'), path.join(duplicate, 'again', 'attestation.json'));
    assert.throws(() => f.lib.verifyAttestations(f.root, duplicate, options), /duplicate smoke attestation/);
  } finally { f.cleanup(); }
});

test('publication gate binds tarballs to build-host tests, smoke runs and the commit', async () => {
  const f = await fixture();
  try {
    const { verifyTestedArtifacts } = await import('../../scripts/verify_tested_artifacts.mjs');
    const manifest = f.lib.verifyRelease(f.root, f.release, { commit: COMMIT });
    const prechecks = fs.mkdtempSync(path.join(f.root, 'prechecks-'));
    const payload = f.lib.packagePayload(f.root);
    const write = (change = () => {}) => {
      for (const item of manifest.platforms) {
        fs.mkdirSync(path.join(prechecks, `precheck-${item.target}`), { recursive: true });
        const tested = { target: item.target, version: '0.1.0', commit: COMMIT, nativeSha256: item.nativeSha256, payload };
        change(tested);
        fs.writeFileSync(path.join(prechecks, `precheck-${item.target}`, 'manifest.json'), JSON.stringify(tested));
      }
    };
    write();
    const gate = () => verifyTestedArtifacts(f.root, { release: f.release, attestations: attestations(f, manifest), prechecks, commit: COMMIT });
    assert.deepEqual(gate(), { targets: 4, smoke: 20 });
    write(tested => { tested.nativeSha256 = '2'.repeat(64); });
    assert.throws(gate, /release binary differs/);
    write(tested => { tested.commit = 'c'.repeat(40); });
    assert.throws(gate, /different commit/);
    write(tested => { tested.commit = null; });
    assert.throws(gate, /different commit/);
    write(tested => { tested.version = '0.0.9'; });
    assert.throws(gate, /tested version changed/);
    write();
    fs.writeFileSync(path.join(f.root, 'README.md'), 'readme\r\n');
    assert.deepEqual(gate(), { targets: 4, smoke: 20 });
    fs.writeFileSync(path.join(f.root, 'README.md'), 'changed readme\n');
    assert.throws(gate, /payload differs/);
  } finally { f.cleanup(); }
});

test('publication uploads only verified tarballs, platforms before the root, and is idempotent', async () => {
  const f = await fixture();
  const log = path.join(f.root, 'npm.log');
  const fake = path.join(f.root, 'fake-npm.cjs');
  fs.writeFileSync(fake, `
    const fs = require('node:fs');
    const argv = process.argv.slice(2);
    fs.appendFileSync(${JSON.stringify(log)}, JSON.stringify(argv) + '\\n');
    const published = JSON.parse(fs.existsSync(${JSON.stringify(log + '.published')}) ? fs.readFileSync(${JSON.stringify(log + '.published')}, 'utf8') : '{}');
    if (argv[0] === 'view') { const integrity = published[argv[1].replace(/@0\\.1\\.0$/, '')]; if (integrity) process.stdout.write(integrity + '\\n'); process.exit(integrity ? 0 : 1); }
  `);
  const previous = process.env.npm_execpath;
  process.env.npm_execpath = fake;
  try {
    const manifest = f.lib.verifyRelease(f.root, f.release, { commit: COMMIT });
    const calls = () => fs.readFileSync(log, 'utf8').trim().split('\n').map(line => JSON.parse(line));
    const publishes = () => calls().filter(call => call[0] === 'publish');
    assert.deepEqual(f.lib.publishOrder(manifest).map(item => item.name), ['narrative-lens-darwin-x64', 'narrative-lens-darwin-arm64', 'narrative-lens-linux-x64-gnu', 'narrative-lens-win32-x64-msvc', 'narrative-lens']);

    f.lib.publishRelease(f.root, f.release, { commit: COMMIT, provenance: true });
    assert.deepEqual(publishes().map(call => path.basename(call[1])), f.lib.publishOrder(manifest).map(item => item.filename));
    for (const call of publishes()) {
      assert.ok(path.isAbsolute(call[1]) && call[1].endsWith('.tgz'), 'publishes a tarball, never a directory');
      assert.ok(call.includes('--ignore-scripts') && call.includes('--provenance') && !call.includes('--dry-run'));
    }

    // A rerun after a partial failure skips identical packages and refuses divergent ones.
    fs.writeFileSync(log + '.published', JSON.stringify(Object.fromEntries(f.lib.publishOrder(manifest).map(item => [item.name, item.integrity]))));
    fs.rmSync(log);
    const rerun = f.lib.publishRelease(f.root, f.release, { commit: COMMIT, dryRun: true });
    assert.ok(rerun.every(item => item.skipped));
    assert.equal(publishes().length, 0);
    fs.writeFileSync(log + '.published', JSON.stringify({ 'narrative-lens-darwin-x64': 'sha512-different' }));
    assert.throws(() => f.lib.publishRelease(f.root, f.release, { commit: COMMIT, dryRun: true }), /already published with different content/);

    // Nothing is uploaded when verification fails.
    fs.rmSync(log);
    fs.rmSync(log + '.published');
    fs.appendFileSync(path.join(f.release, 'narrative-lens-0.1.0.tgz'), 'x');
    assert.throws(() => f.lib.publishRelease(f.root, f.release, { commit: COMMIT }));
    assert.equal(fs.existsSync(log), false);
  } finally {
    if (previous === undefined) delete process.env.npm_execpath; else process.env.npm_execpath = previous;
    f.cleanup();
  }
});

test('release workflow publishes only after tested assembly and every smoke run', async () => {
  const { load } = await import('js-yaml');
  const read = name => fs.readFileSync(path.join(__dirname, '..', '..', '.github', 'workflows', name), 'utf8');
  const release = load(read('release.yml'));
  const assurance = load(read('platform-assurance.yml'));
  const pkg = require('../../package.json');
  const { supportedNodeMajors } = await scripts();
  assert.deepEqual(Object.keys(release.on), ['workflow_dispatch']);
  assert.deepEqual(Object.keys(assurance.on).sort(), ['workflow_call', 'workflow_dispatch']);
  assert.deepEqual(release.permissions, { contents: 'read' });
  assert.deepEqual(assurance.permissions, { contents: 'read' });
  assert.deepEqual(Object.keys(release.jobs), ['validate', 'assurance', 'publish']);
  assert.deepEqual(Object.keys(assurance.jobs), ['build', 'assemble', 'smoke']);
  assert.equal(release.jobs.assurance.uses, './.github/workflows/platform-assurance.yml');
  assert.equal(release.jobs.assurance.needs, 'validate');
  assert.deepEqual([...release.jobs.publish.needs].sort(), ['assurance', 'validate']);
  const jobs = assurance.jobs;
  assert.equal(jobs.assemble.needs, 'build');
  assert.equal(jobs.smoke.needs, 'assemble');

  // GitHub's matrix semantics: dimensions form the product; an `include` entry is merged into every
  // combination it does not contradict, or becomes its own job when it fits none (or when there are no dimensions).
  const expand = ({ include = [], ...dimensions }) => {
    const names = Object.keys(dimensions);
    let combinations = names.length ? [{}] : [];
    for (const name of names) combinations = combinations.flatMap(base => dimensions[name].map(value => ({ ...base, [name]: value })));
    const result = combinations.map(item => ({ ...item }));
    for (const entry of include) {
      const fits = result.filter(item => names.every(name => !(name in entry) || entry[name] === item[name]));
      if (fits.length && names.length) for (const item of fits) Object.assign(item, entry);
      else result.push({ ...entry });
    }
    return result;
  };
  const byTarget = Object.fromEntries(Object.entries({ 'x86_64-apple-darwin': /^macos-.*intel/, 'aarch64-apple-darwin': /^macos-(latest|\d+)$/, 'x86_64-unknown-linux-gnu': /^ubuntu-/, 'x86_64-pc-windows-msvc': /^windows-/ }));
  const targets = [...pkg.napi.targets].sort();
  const build = expand(jobs.build.strategy.matrix);
  assert.deepEqual(build.map(item => item.target).sort(), targets);
  const smoke = expand(jobs.smoke.strategy.matrix);
  assert.equal(smoke.length, targets.length * supportedNodeMajors.length, 'every target must run on every supported Node line');
  assert.deepEqual(smoke.map(item => `${item.target}|${item.node}`).sort(), targets.flatMap(target => supportedNodeMajors.map(node => `${target}|${node}`)).sort());
  for (const item of [...build, ...smoke]) assert.match(item.host, byTarget[item.target], `${item.target} runs on the wrong runner: ${item.host}`);
  assert.deepEqual([...new Set(build.map(item => item.host))].length, targets.length, 'one runner per target');
  assert.equal(jobs.build.strategy['fail-fast'], false);
  assert.equal(jobs.smoke.strategy['fail-fast'], false);
  assert.match(release.jobs.publish.if, /refs\/heads\/main/);
  assert.equal(release.jobs.publish.environment, 'npm-release');

  const everyJob = [...Object.entries(release.jobs), ...Object.entries(jobs)];
  assert.deepEqual(everyJob.filter(([, job]) => job.permissions?.['id-token'] === 'write').map(([name]) => name), ['publish']);
  for (const name of ['release.yml', 'platform-assurance.yml', 'publish-crate.yml']) {
    const uses = [...read(name).matchAll(/uses:\s+(\S+)/g)].map(match => match[1]).filter(value => !value.startsWith('./'));
    for (const value of uses) assert.match(value, /@[0-9a-f]{40}$/, `${value} must be pinned to a commit`);
  }

  const steps = release.jobs.publish.steps;
  const index = pattern => steps.findIndex(step => pattern.test(step.run ?? ''));
  const gate = index(/verify_tested_artifacts\.mjs/);
  const publish = index(/release_artifacts\.mjs publish/);
  assert.ok(gate >= 0 && publish > gate, 'the publication gate must run before publishing');
  const text = JSON.stringify([release, assurance]);
  assert.ok(!/npm publish/.test(text), 'publication goes through the verified-tarball publisher');
  assert.ok(!/napi prepublish(?![^"]*--skip-optional-publish)/.test(text), 'napi prepublish must not publish');
  assert.match(steps[publish].run, /--provenance/);
  const smokeRun = jobs.smoke.steps.map(step => step.run ?? '').join('\n');
  assert.match(smokeRun, /args=\(--release release-packages --attest "\$ATTESTATION"\)/);
  assert.match(smokeRun, /\[ "\$NODE_MAJOR" = 24 \]; then args\+=\(--types\)/);
  assert.match(smokeRun, /npm run test:package -- "\$\{args\[@\]\}"/);
  const uploads = jobs.smoke.steps.filter(step => /upload-artifact/.test(step.uses ?? ''));
  assert.match(uploads[0].with.name, /^attestation-/);

  const crate = load(read('publish-crate.yml'));
  const crateSteps = crate.jobs.publish.steps;
  assert.equal(crateSteps.find(step => /actions\/checkout/.test(step.uses)).with['persist-credentials'], false);
  const tagCheck = crateSteps.findIndex(step => /GITHUB_REF_NAME|release\.tag_name/.test(JSON.stringify(step)) && /Cargo\.toml|cargo metadata/.test(step.run ?? ''));
  const cargoPublish = crateSteps.findIndex(step => /cargo publish --locked$/.test(step.run ?? ''));
  assert.ok(tagCheck >= 0 && cargoPublish > tagCheck, 'the release tag must be checked against Cargo.toml before cargo publish');
});
