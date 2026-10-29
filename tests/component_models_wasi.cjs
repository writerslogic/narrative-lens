const { readFileSync } = require('node:fs');
const { resolve } = require('node:path');
const { WASI } = require('node:wasi');

async function main() {
  if (!process.argv[2]) throw new Error('Usage: node tests/component_models_wasi.cjs <component_models.wasm>');
  const artifact = resolve(process.argv[2]);
  const repository = resolve(__dirname, '..');
  const wasi = new WASI({
    version: 'preview1',
    returnOnExit: true,
    args: [artifact, '--test-threads=1', ...process.argv.slice(3)],
    preopens: { [repository]: repository },
  });
  const module = await WebAssembly.compile(readFileSync(artifact));
  const instance = await WebAssembly.instantiate(module, wasi.getImportObject());
  process.exitCode = wasi.start(instance);
}

main().catch(error => { console.error(error); process.exitCode = 1; });
