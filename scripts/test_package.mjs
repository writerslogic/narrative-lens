import assert from 'node:assert/strict';
import { mkdtempSync, mkdirSync, readdirSync, readFileSync, copyFileSync, writeFileSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { basename, dirname, join, resolve } from 'node:path';
import { createRequire } from 'node:module';
import { spawn, spawnSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';
import { digest, packagePayload, platform, releasePlan, stageNotices, startRegistry, verifyArtifacts, verifyRelease } from './release_artifacts.mjs';

const root = fileURLToPath(new URL('..', import.meta.url));
const args = process.argv.slice(2);
function option(name) {
  const index = args.indexOf(name);
  if (index < 0) return undefined;
  assert.ok(args[index + 1] && !args[index + 1].startsWith('--'), `${name} requires a value`);
  return args[index + 1];
}
// A stray override would make the loader skip the installed platform package.
function environment() {
  const env = { ...process.env, npm_config_audit: 'false', npm_config_fund: 'false', NAPI_RS_ENFORCE_VERSION_CHECK: '1' };
  delete env.NAPI_RS_NATIVE_LIBRARY_PATH;
  return env;
}
function run(command, argv, cwd) {
  const result = spawnSync(command, argv, { cwd, encoding: 'utf8', env: environment() });
  if (result.error) throw result.error;
  assert.equal(result.status, 0, `${basename(command)} failed: ${(result.stderr || result.stdout).slice(-6000)}`);
  return result.stdout;
}
function npm(argv, cwd) {
  assert.ok(process.env.npm_execpath, 'run this check through npm run test:package');
  return run(process.execPath, [process.env.npm_execpath, ...argv], cwd);
}
// The registry lives in this process, so installs against it must not block the event loop.
function npmAsync(argv, cwd) {
  assert.ok(process.env.npm_execpath, 'run this check through npm run test:package');
  return new Promise((done, fail) => {
    const child = spawn(process.execPath, [process.env.npm_execpath, ...argv], { cwd, env: environment() });
    let output = '';
    child.stdout.on('data', chunk => { output += chunk; });
    child.stderr.on('data', chunk => { output += chunk; });
    child.on('error', fail);
    child.on('close', status => status === 0 ? done(output) : fail(new Error(`npm ${argv[0]} failed: ${output.slice(-6000)}`)));
  });
}
function glibc() {
  if (process.platform !== 'linux') return undefined;
  process.report.excludeNetwork = true;
  return process.report.getReport().header.glibcVersionRuntime ? 'glibc' : 'musl';
}
function hostTarget() {
  const target = ({ darwin: { arm64: 'aarch64-apple-darwin', x64: 'x86_64-apple-darwin' }, linux: { x64: 'x86_64-unknown-linux-gnu' }, win32: { x64: 'x86_64-pc-windows-msvc' } })[process.platform]?.[process.arch];
  assert.ok(target, `${process.platform}/${process.arch} is not a declared release target`);
  assert.ok(glibc() !== 'musl', 'Linux releases target glibc only');
  return target;
}
function smokeSource(suffix) {
  return `
    const assert = require('node:assert/strict');
    const fs = require('node:fs');
    const path = require('node:path');
    assert.equal(process.env.NAPI_RS_NATIVE_LIBRARY_PATH, undefined);
    const api = require('narrative-lens');
    const installed = path.dirname(require.resolve('narrative-lens'));
    assert.ok(!fs.readdirSync(installed).some(file => file.endsWith('.node')), 'root tarball must load the platform package');
    const platformRoot = path.dirname(require.resolve('narrative-lens-' + ${JSON.stringify(suffix)} + '/package.json'));
    assert.equal(fs.readFileSync(path.join(platformRoot, 'THIRD_PARTY_NOTICES'), 'utf8'), fs.readFileSync(path.join(installed, 'THIRD_PARTY_NOTICES'), 'utf8'));
    assert.equal(fs.readFileSync(path.join(platformRoot, 'LICENSE'), 'utf8'), fs.readFileSync(path.join(installed, 'LICENSE'), 'utf8'));
    assert.equal(require(path.join(platformRoot, 'package.json')).version, require(path.join(installed, 'package.json')).version);
    const text = '😀 She saw saw the bird.';
    const report = api.analyzeManuscript({ documentId: 'package', scenes: [{ id: 'first', text }], configuration: { rules: ['NL-GRAMMAR-REPETITION'] } });
    assert.equal(report.findings.length, 1);
    for (const evidence of report.findings[0].evidence) assert.equal(text.slice(evidence.range.utf16Start, evidence.range.utf16End), evidence.quote);
    assert.ok('characterAt' in api.trackWorldState(['Alice was in the cellar.'], ['Alice']).snapshots[0].facts[0]);
    assert.deepEqual(api.trackWorldStateCurrent(['Alice was in the cellar.'],['Alice']).facts, api.trackWorldState(['Alice was in the cellar.'],['Alice']).snapshots[0].facts);
    const source = {sceneId:'first',utf16Start:0,utf16End:3,quote:'The'};
    const authored = api.analyzeAuthorKnowledge({documentId:'authored',scenes:[{id:'first',text:'The key is hidden.'}]},[{id:'key',description:'The key is hidden',truth:'unknown',evidence:source}],[{id:'reader',sequence:1,propositionId:'key',action:{kind:'uncertain',actor:{kind:'reader'}},evidence:source}]);
    assert.equal(authored.states[0].classification,'uncertain');
    import('narrative-lens').then(async esm => assert.deepEqual(await esm.analyzeManuscriptAsync({ documentId: 'package', scenes: [{ id: 'first', text }], configuration: { rules: ['NL-GRAMMAR-REPETITION'] } }), report)).catch(error => { console.error(error); process.exitCode = 1; });
  `;
}
// A real ESM consumer: static named imports through the `import` export condition.
const esmSource = `
  import assert from 'node:assert/strict';
  import { createRequire } from 'node:module';
  import * as esm from 'narrative-lens';
  import { analyzeManuscript, analyzeManuscriptAsync } from 'narrative-lens';
  const cjs = createRequire(import.meta.url)('narrative-lens');
  assert.deepEqual(Object.keys(cjs).filter(name => !name.startsWith('__') && !(name in esm)), [], 'the ESM entry must export everything the CommonJS entry does');
  const input = { documentId: 'esm', scenes: [{ id: 'first', text: 'She saw saw the bird.' }], configuration: { rules: ['NL-GRAMMAR-REPETITION'] } };
  assert.equal(analyzeManuscript(input).findings.length, 1);
  assert.deepEqual(await analyzeManuscriptAsync(input), analyzeManuscript(input));
`;
function smoke(consumer, suffix) {
  run(process.execPath, ['-e', smokeSource(suffix)], consumer);
  writeFileSync(join(consumer, 'esm-consumer.mjs'), esmSource);
  run(process.execPath, ['esm-consumer.mjs'], consumer);
}
function consumerProject(parent, name) {
  const consumer = join(parent, name);
  mkdirSync(consumer);
  writeFileSync(join(consumer, 'package.json'), '{"private":true}');
  return consumer;
}
function typeCheck(consumer) {
  writeFileSync(join(consumer, 'consumer.mts'), "import { analyzeManuscript, trackWorldState } from 'narrative-lens';\nconst count: number = analyzeManuscript({documentId:'typed',scenes:[]}).summary.totalFindings;\nconst fact = trackWorldState(['Alice was in the cellar.'],['Alice']).snapshots[0].facts[0];\nif ('characterAt' in fact) { const location: string = fact.characterAt.location; void location; }\nvoid count;\n");
  writeFileSync(join(consumer, 'consumer.cts'), "import api = require('narrative-lens');\nconst count: number = api.analyzeManuscript({documentId:'typed',scenes:[]}).summary.totalFindings;\nvoid count;\n");
  writeFileSync(join(consumer, 'tsconfig.json'), JSON.stringify({ compilerOptions: { strict: true, noEmit: true, target: 'es2022', module: 'nodenext', moduleResolution: 'nodenext', types: [] }, files: ['consumer.mts', 'consumer.cts'] }));
  run(process.execPath, [join(root, 'node_modules', 'typescript', 'bin', 'tsc'), '-p', consumer], consumer);
}
function installedPlatforms(consumer, name) {
  const modules = join(consumer, 'node_modules');
  return readdirSync(modules).filter(entry => entry.startsWith(`${name}-`)).sort();
}

const temporary = mkdtempSync(join(tmpdir(), 'narrative-lens-package-'));
try {
  const plan = releasePlan(root);
  const releaseOption = option('--release');
  const target = releaseOption ? hostTarget() : option('--target') || hostTarget();
  const host = platform(target);
  const withTypes = args.includes('--types');
  let archives;
  let release;
  if (releaseOption) {
    // Install exactly the tarballs that the publish job will upload.
    const commit = option('--commit') ?? process.env.GITHUB_SHA;
    const directory = resolve(releaseOption);
    release = verifyRelease(root, directory, { commit });
    const built = release.platforms.find(item => item.target === target);
    assert.ok(built, `release does not contain ${target}`);
    archives = [join(directory, release.root.filename), join(directory, built.filename)];
    release.directory = directory;
  } else {
    const staging = join(temporary, 'package');
    mkdirSync(staging);
    for (const file of [...plan.pkg.files, 'README.md', 'LICENSE']) copyFileSync(join(root, file), join(staging, file));
    // Each native host stages its own target; the publishing gate requires all targets.
    writeFileSync(join(staging, 'package.json'), JSON.stringify({ ...plan.pkg, napi: { ...plan.pkg.napi, targets: [target] } }, null, 2));
    const require = createRequire(import.meta.url);
    const cliPackage = require.resolve('@napi-rs/cli/package.json');
    const cliMetadata = JSON.parse(readFileSync(cliPackage, 'utf8'));
    const cli = join(dirname(cliPackage), typeof cliMetadata.bin === 'string' ? cliMetadata.bin : cliMetadata.bin.napi);
    run(process.execPath, [cli, 'create-npm-dirs', '--cwd', staging], root);
    const filename = `${plan.pkg.napi.binaryName}.${host.suffix}.node`;
    copyFileSync(join(root, filename), join(staging, 'npm', host.suffix, filename));
    stageNotices(staging, [target]);
    run(process.execPath, [cli, 'prepublish', '--cwd', staging, '--skip-optional-publish', '--no-gh-release', '-t', 'npm'], root);
    verifyArtifacts(staging, [target]);
    const output = resolve(option('--output') || join(root, 'target', 'package-smoke', target));
    mkdirSync(output, { recursive: true });
    archives = [staging, join(staging, 'npm', host.suffix)].map(directory => {
      const packed = JSON.parse(npm(['pack', '--ignore-scripts', '--json', '--pack-destination', output], directory))[0];
      return join(output, packed.filename);
    });
    const manifest = { target, version: plan.pkg.version, commit: process.env.GITHUB_SHA ?? null, nativeSha256: digest(join(root, filename)), payload: packagePayload(root), archives: archives.map(path => ({ filename: basename(path), sha256: digest(path) })) };
    writeFileSync(join(output, 'manifest.json'), JSON.stringify(manifest, null, 2) + '\n');
  }
  assert.equal(archives.length, 2, 'expected the root and one platform archive');

  const consumer = consumerProject(temporary, 'consumer');
  npm(['install', '--ignore-scripts', '--offline', '--omit=optional', '--no-package-lock', ...archives], consumer);
  smoke(consumer, host.suffix);

  const attestation = { schema: 1, kind: 'smoke', types: false, registryResolution: false, checks: ['cjs', 'esm', 'manuscript', 'notices', 'version-lock'] };
  if (release) {
    // Resolve the platform package through optionalDependencies, as a registry user does.
    const registry = await startRegistry(release.directory);
    try {
      const cache = join(temporary, 'registry-cache');
      const install = (folder, extra = []) => npmAsync(['install', `${release.name}@${release.version}`, '--registry', registry.url, '--cache', cache, '--ignore-scripts', '--no-package-lock', '--no-audit', '--no-fund', ...extra], folder);
      const registryConsumer = consumerProject(temporary, 'registry');
      await install(registryConsumer);
      assert.deepEqual(installedPlatforms(registryConsumer, release.name), [`${release.name}-${host.suffix}`], 'the registry must resolve exactly the host platform package');
      smoke(registryConsumer, host.suffix);
      if (withTypes) typeCheck(registryConsumer);
      // Installation selection only: these binaries cannot execute on this host.
      const selection = [];
      for (const item of plan.packages.filter(candidate => candidate.target !== target)) {
        const folder = consumerProject(temporary, `select-${item.host.suffix}`);
        await install(folder, ['--os', item.host.os, '--cpu', item.host.cpu, ...item.host.libc ? ['--libc', item.host.libc] : []]);
        assert.deepEqual(installedPlatforms(folder, release.name), [item.name], `${item.target} must select only its own platform package`);
        selection.push({ target: item.target, installed: [item.name] });
      }
      for (const unsupported of [['linux', 'arm64', 'glibc'], ['linux', 'x64', 'musl'], ['win32', 'arm64', undefined]]) {
        const folder = consumerProject(temporary, `unsupported-${unsupported.join('-')}`);
        await install(folder, ['--os', unsupported[0], '--cpu', unsupported[1], ...unsupported[2] ? ['--libc', unsupported[2]] : []]);
        assert.deepEqual(installedPlatforms(folder, release.name), [], `${unsupported.filter(Boolean).join('/')} must not receive a platform binary`);
        selection.push({ target: unsupported.filter(Boolean).join('/'), installed: [] });
      }
      Object.assign(attestation, { registryResolution: true, selection });
    } finally {
      await registry.close();
    }
    const built = release.platforms.find(item => item.target === target);
    Object.assign(attestation, {
      commit: release.commit,
      version: release.version,
      target,
      platform: process.platform,
      arch: process.arch,
      libc: glibc(),
      node: process.versions.node,
      npm: npm(['--version'], temporary).trim(),
      releaseManifestSha256: digest(join(release.directory, 'release-manifest.json')),
      nativeSha256: built.nativeSha256,
      archives: [{ filename: release.root.filename, sha256: release.root.sha256 }, { filename: built.filename, sha256: built.sha256 }],
      types: withTypes,
    });
    if (attestation.libc === undefined) delete attestation.libc;
    const attest = option('--attest');
    if (attest) {
      mkdirSync(dirname(resolve(attest)), { recursive: true });
      writeFileSync(resolve(attest), JSON.stringify(attestation, null, 2) + '\n');
    }
  } else {
    typeCheck(consumer);
  }
  process.stdout.write(`Installed package CJS, ESM and manuscript smoke passed on ${process.platform}/${process.arch} Node ${process.version}${release ? ' with registry resolution' : ''}\n`);
} finally {
  rmSync(temporary, { recursive: true, force: true });
}
