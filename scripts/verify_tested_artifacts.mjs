import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { digest, packagePayload, releasePlan, verifyAttestations, verifyRelease } from './release_artifacts.mjs';

/**
 * Publication gate. The assembled tarballs, the binaries installed by the build hosts,
 * and one smoke attestation per target and supported Node line must all describe the
 * same commit and the same bytes.
 */
export function verifyTestedArtifacts(root, { release, attestations, prechecks, commit }) {
  const plan = releasePlan(root);
  const manifest = verifyRelease(root, release, { commit });
  const payload = packagePayload(root);
  for (const item of manifest.platforms) {
    const tested = JSON.parse(readFileSync(join(prechecks, `precheck-${item.target}`, 'manifest.json'), 'utf8'));
    assert.equal(tested.target, item.target, 'tested target changed');
    assert.equal(tested.version, plan.pkg.version, 'tested version changed');
    assert.equal(tested.commit, commit, 'build host tested a different commit');
    assert.equal(tested.nativeSha256, item.nativeSha256, 'release binary differs from the installed/tested binary');
    assert.deepEqual(tested.payload, payload, 'release payload differs from the installed/tested package');
  }
  const smoke = verifyAttestations(root, attestations, { commit, manifest, manifestSha256: digest(join(release, 'release-manifest.json')) });
  return { targets: manifest.platforms.length, smoke };
}

function option(args, name) {
  const index = args.indexOf(name);
  assert.ok(index >= 0 && args[index + 1] && !args[index + 1].startsWith('--'), `${name} is required`);
  return resolve(args[index + 1]);
}

if (process.argv[1] && resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  const root = fileURLToPath(new URL('..', import.meta.url));
  const args = process.argv.slice(2);
  try {
    const commitIndex = args.indexOf('--commit');
    const commit = commitIndex >= 0 ? args[commitIndex + 1] : process.env.GITHUB_SHA;
    const result = verifyTestedArtifacts(root, { release: option(args, '--release'), attestations: option(args, '--attestations'), prechecks: option(args, '--prechecks'), commit });
    process.stdout.write(`All ${result.targets} release targets and ${result.smoke} smoke attestations match commit ${commit}\n`);
  } catch (error) {
    process.stderr.write(`Publication gate failed: ${error.message}\n`);
    process.exitCode = 1;
  }
}
