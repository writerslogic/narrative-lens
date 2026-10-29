// Original fiction and engineering annotations; no human editorial validation.
import { writeFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
const cases=[];
function scene(id,text,intentions=[]){return {id,text,intentions};}
function add(split,family,id,scenes,rules,expected,configuration={},knownCharacters=['Zoë']){
  const manuscript={documentId:id,scenes,knownCharacters,configuration:{rules,...configuration}};
  cases.push({id,split,family,annotationOrigin:'OpenAI Codex; original synthetic fiction; agent-authored labels; no human editorial review',rationale:family,manuscript,expected});
}
const gram=['NL-GRAMMAR-REPETITION','NL-GRAMMAR-WHITESPACE'];
const exp=(ruleId,sceneId,quote,occurrence=0,related=[])=>({ruleId,sceneId,quote,occurrence,related});
for(const [split,name,place,genre] of [['calibration','Zoë','harbor','Mystery'],['held_out','Mara','observatory','SciFi']]){
  const pre=split==='calibration'?'c':'h';
  add(split,`${pre}-copyedit-two-actions`,`${pre}-mechanics-mixed`,[scene('opening',`${name} saw saw the sealed door. A  bell rang beyond the ${place}.`) ],gram,[exp(gram[0],'opening','saw',1),exp(gram[1],'opening',' ',7)]);
  add(split,`${pre}-different-repetition-contexts`,`${pre}-duplicates`,[scene('opening',`${name} heard heard the bell. The child heard heard the sea.`)],gram,[exp(gram[0],'opening','heard',1),exp(gram[0],'opening','heard',3)]);
  add(split,`${pre}-clean-mechanics`,`${pre}-clean`,[scene('opening',`${name} counted the boats at the ${place}. Each had a light at its mast.`)],gram,[]);
  add(split,`${pre}-deliberate-emphasis`,`${pre}-intentional-repetition`,[scene('opening',`Never never open that door, ${name} whispered.`,['repetition'])],gram,[]);
  add(split,`${pre}-punctuated-repetition`,`${pre}-rhetorical-punctuation`,[scene('opening',`Run, run, ${name} said. The child stayed by the ${place}.`)],gram,[]);
  add(split,`${pre}-unicode-mechanics`,`${pre}-unicode`,[scene('opening',`😀 雪. ${name} came came back to the ${place}.`) ],gram,[exp(gram[0],'opening','came',1)]);
  add(split,`${pre}-typo-versus-author-dictionary`,`${pre}-dictionary`,[scene('opening',`${name} saw a qwick bird beside the wyrmlet.`)],['NL-GRAMMAR-SPELLING'],[exp('NL-GRAMMAR-SPELLING','opening','qwick')],{dictionary:['wyrmlet']});
  add(split,`${pre}-dialect`,`${pre}-dialect`,[scene('opening',`Keep walkin till the moon is high, ${name} said.`,['dialect'])],['NL-GRAMMAR-SPELLING'],[]);
  add(split,`${pre}-editorial-terms`,`${pre}-lexicon`,[scene('opening',`${name} suddenly saw the suddenly bright lights near the ${place}.`) ],['NL-PROSE-LEXICON'],[exp('NL-PROSE-LEXICON','opening','suddenly'),exp('NL-PROSE-LEXICON','opening','suddenly',1)],{lexicon:['suddenly']});
  add(split,`${pre}-whole-word-boundaries`,`${pre}-lexicon-boundary`,[scene('opening',`The café was almost empty. ${name} waited by the ${place}.`)],['NL-PROSE-LEXICON'],[],{lexicon:['cafe','most']});
  add(split,`${pre}-locked-location`,`${pre}-location`,[scene('before',`${name} was in the cellar. Someone locked the door.`),scene('after',`${name} entered the garden.`)],['NL-CONTINUITY-LOCKED'],[exp('NL-CONTINUITY-LOCKED','after',`${name} entered the garden.`,0,[{sceneId:'before',quote:`${name} was in the cellar. Someone locked the door.`}])],{},[name]);
  add(split,`${pre}-explicit-release`,`${pre}-release`,[scene('before',`${name} was in the cellar. Someone locked the door.`),scene('after',`${name} escaped from the cellar. ${name} entered the garden.`)],['NL-CONTINUITY-LOCKED'],[],{},[name]);
  add(split,`${pre}-intentional-physical-exception`,`${pre}-supernatural`,[scene('before',`${name} was in the cellar. Someone locked the door.`),scene('after',`${name} entered the garden.`,['supernatural'])],['NL-CONTINUITY-LOCKED'],[],{},[name]);
  add(split,`${pre}-stable-attribute`,`${pre}-eye-conflict`,[scene('before',`${name} had blue eyes.`),scene('after',`${name} had brown eyes.`)],['NL-CONTINUITY-ATTRIBUTE'],[exp('NL-CONTINUITY-ATTRIBUTE','after',`${name} had brown eyes.`,0,[{sceneId:'before',quote:`${name} had blue eyes.`}])],{},[name]);
  add(split,`${pre}-dialogue-is-not-fact`,`${pre}-quoted-eye`,[scene('before',`${name} had blue eyes.`),scene('after',`😀 “${name} had brown eyes.”`)],['NL-CONTINUITY-ATTRIBUTE'],[],{},[name]);
  add(split,`${pre}-spelling-variant-colors`,`${pre}-grey-gray`,[scene('before',`${name} had grey eyes.`),scene('after',`${name} had gray eyes.`)],['NL-CONTINUITY-ATTRIBUTE'],[],{},[name]);
  add(split,`${pre}-object-change`,`${pre}-object`,[scene('before',split==='calibration'?'The key was on the desk.':'The key was in the drawer.'),scene('after',split==='calibration'?'The key was on the shelf.':'The key was on the table.')],['NL-CONTINUITY-OBJECT'],[exp('NL-CONTINUITY-OBJECT','after',split==='calibration'?'The key was on the shelf.':'The key was on the table.',0,[{sceneId:'before',quote:split==='calibration'?'The key was on the desk.':'The key was in the drawer.'}])]);
  add(split,`${pre}-explicit-object-transfer`,`${pre}-object-transfer`,[scene('before','The book was in the cabin.'),scene('bridge',`${name} carried the book to the bridge.`),scene('after','The book was on the bridge.')],['NL-CONTINUITY-OBJECT'],[],{},[name]);
  add(split,`${pre}-empty-input`,`${pre}-empty`,[scene('opening','')],['NL-READABILITY-COMPLEX','NL-PACING-LONG'],[]);
  const complex=split==='calibration'?'Institutional representatives reconsidered constitutional obligations while investigating interplanetary communication and bureaucratic responsibility. ':'Extraterrestrial investigators documented irreversible environmental deterioration, contradictory philosophical interpretations, and intergenerational institutional reconstruction. ';
  add(split,`${pre}-dense-exposition`,`${pre}-readability`,[scene('opening',complex.repeat(12))],['NL-READABILITY-COMPLEX'],[exp('NL-READABILITY-COMPLEX','opening',null)],{genre});
  add(split,`${pre}-intentional-dense-narrator`,`${pre}-readability-intention`,[scene('opening',complex.repeat(12),['complex_prose'])],['NL-READABILITY-COMPLEX'],[],{genre});
  const long=`${name} followed the road past the ${place} and over the hill and through the trees and down to the field where the birds called from the branches and the light fell on the water and the path turned toward the town and the river shone beside the old stone bridge and the gate stood open by the house and the clouds moved across the sky and the wind bent the grass beside the wall and the bell rang in the tower while the ferry left the far bank and the first stars came out above the dark fields beyond the old house where a child waited by the fire.`;
  add(split,`${pre}-unbroken-sentence`,`${pre}-pacing`,[scene('opening',long)],['NL-PACING-LONG'],[exp('NL-PACING-LONG','opening',null)],{genre});
  add(split,`${pre}-intentional-breathless-sentence`,`${pre}-pacing-intention`,[scene('opening',long,['long_sentences'])],['NL-PACING-LONG'],[],{genre});
  const structure=[scene('setup',`${name} found a letter at the ${place}.`),scene('middle',`${name} followed the trail across the sea.`),scene('ending',`${name} came home with the missing child.`)];
  for(const [i,s]of structure.entries())s.purpose=['setup inciting incident','confrontation midpoint','climax resolution'][i];
  add(split,`${pre}-declared-three-act-arc`,`${pre}-structure`,structure,['NL-STRUCTURE-MISSING-BEAT'],[]);
  const flat=[scene('one',`The clock at the ${place} stopped.`),scene('two',`The clock at the ${place} stopped.`),scene('three',`The clock at the ${place} stopped.`)];
  add(split,`${pre}-missing-conventional-beats`,`${pre}-flat`,flat,['NL-STRUCTURE-MISSING-BEAT'],[{ruleId:'NL-STRUCTURE-MISSING-BEAT',sceneId:null,quote:null,occurrence:0,related:[],messageContains:'Climax'}]);
  add(split,`${pre}-intentional-cyclical-form`,`${pre}-nonlinear`,flat.map(s=>({...s,intentions:['nonlinear']})),['NL-STRUCTURE-MISSING-BEAT'],[]);
}
for(const c of cases){c.exhaustive=!c.id.endsWith('-flat');for(const expected of c.expected){
  if(expected.quote!==null){const source=c.manuscript.scenes.find(s=>s.id===expected.sceneId).text;let start=-1;for(let i=0;i<=expected.occurrence;i++)start=source.indexOf(expected.quote,start+1);if(start<0)throw Error(`${c.id}: missing annotation`);expected.range={byteStart:Buffer.byteLength(source.slice(0,start)),byteEnd:Buffer.byteLength(source.slice(0,start+expected.quote.length)),utf16Start:start,utf16End:start+expected.quote.length};}
}}
const sha=value=>createHash('sha256').update(value).digest('hex');
for(const c of cases)c.textSha256=sha(c.manuscript.scenes.map(s=>s.text).join('\n\x1e\n'));
const corpus={version:'finding-level-fiction-v2.1',annotationOrigin:'OpenAI Codex agent-authored original fiction and annotations; no human editorial review',splitPolicy:'Nonempty complete passage texts are disjoint across splits; empty input appears in both as an unscored boundary case. Related synthetic templates remain. These are engineering cases, not independent editorial samples. Calibration may only use calibration labels.',cases};
writeFileSync(new URL('../tests/fixtures/evaluation-v2.json',import.meta.url),JSON.stringify(corpus,null,2)+'\n');
