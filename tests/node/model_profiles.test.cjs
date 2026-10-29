const test = require('node:test');
const assert = require('node:assert/strict');
const {mkdtempSync,mkdirSync,readFileSync,writeFileSync,rmSync} = require('node:fs');
const {tmpdir} = require('node:os');
const {join} = require('node:path');
const {createHash} = require('node:crypto');
const api = require('../../index.js');

test('verified Node model loading pins actual bytes and carries identity into reports', async context => {
  const root=mkdtempSync(join(tmpdir(),'narrative-lens-node-model-'));
  try {
    const directory=join(root,'nli-onnx');mkdirSync(directory);
    const model=readFileSync(join(__dirname,'..','fixtures','models','valid-nli.onnx'));
    const tokenizer=Buffer.from(JSON.stringify({version:'1.0',truncation:null,padding:null,added_tokens:[],normalizer:null,pre_tokenizer:null,post_processor:null,decoder:null,model:{type:'WordLevel',vocab:{'[UNK]':0},unk_token:'[UNK]'}}));
    writeFileSync(join(directory,'model.onnx'),model);writeFileSync(join(directory,'tokenizer.json'),tokenizer);
    const digest=value=>createHash('sha256').update(value).digest('hex');
    const manifest={schemaVersion:1,kind:'nli',profile:'hugging_face_nli3_v1',modelSha256:digest(model),tokenizerSha256:digest(tokenizer),maxTokens:512,paddingTokenId:0,labels:['contradiction','neutral','entailment']};
    let identity;
    try { identity=await api.loadVerifiedModel('nli',root,manifest); }
    catch(error) {
      if (/requires an addon built with the onnx feature/.test(error.message)) {
        if (process.env.NARRATIVE_LENS_REQUIRE_NODE_MODELS==='1') throw error;
        context.diagnostic('Default addon explicitly rejects model loading; the onnx addon gate executes this fixture.');
        return;
      }
      throw error;
    }
    assert.equal(identity.verification,'verified_manifest_v1');
    assert.equal(identity.manifest.modelSha256,manifest.modelSha256);
    const manuscript={documentId:'node-pins',scenes:[{id:'before',text:'Alice stayed.'},{id:'after',text:'Alice left.'}],configuration:{rules:['NL-CONTINUITY-NLI'],nli:'fallback'}};
    const report=await api.analyzeManuscriptAsync(manuscript);
    assert.equal(report.findings.length,1);
    assert.deepEqual(report.modelIdentities,[identity]);
    const esm=await import('../../index.mjs');
    assert.deepEqual(await esm.loadVerifiedModel('nli',root,manifest),identity);
    writeFileSync(join(directory,'model.onnx'),'tampered');
    await assert.rejects(api.loadVerifiedModel('nli',root,manifest),/model_digest_mismatch/);
    const failed=await api.analyzeManuscriptAsync(manuscript);
    assert.equal(failed.findings.length,0);
    assert.deepEqual(failed.modelIdentities,[]);
    assert.equal(failed.coverage.find(c=>c.analyzer==='nli').status,'execution_failure');
  } finally {rmSync(root,{recursive:true,force:true});}
});
