// Original synthetic fiction; seeded performance workload, not accuracy labels.
let state = 20261027;
function pick(items) {
  state ^= state << 13; state ^= state >>> 17; state ^= state << 5;
  return items[(state >>> 0) % items.length];
}
const places = ['harbor', 'garden', 'bridge', 'market', 'forest', 'village', 'shore', 'tower', 'valley', 'station'];
const objects = ['map', 'lamp', 'letter', 'clock', 'coat', 'rope', 'basket', 'stone', 'shell', 'bottle'];
const weather = ['rain', 'wind', 'snow', 'mist', 'cold', 'heat', 'light', 'dark'];
const names = ['Maren', 'Adrian', 'Zoë'];
const scenes = Array.from({ length: 160 }, (_, i) => {
  const sentences = [];
  for (let j = 0; j < 45; j++) {
    const name = pick(names), place = pick(places), object = pick(objects), sky = pick(weather);
    sentences.push(pick([
      `${name} crossed the ${place} with the ${object} held close against the ${sky}.`,
      `Beyond the ${place}, the ${sky} covered every trace of the road they had followed.`,
      `The ${object} was all that ${name} had brought from home before the ${sky} came.`,
      `“Keep the ${object},” ${name} said, “until we reach the ${place} and find a safe way home.”`,
      `No one at the ${place} would speak about the ${object}, though ${name} asked twice.`,
      `${name} waited by the ${place} until the ${sky} passed and the last voices fell quiet.`,
      `Under the ${sky}, a child left the ${object} beside the path and walked toward the ${place}.`,
      `For a moment ${name} thought of the ${place}, and of the ${object} still waiting there.`,
    ]));
  }
  if (i % 17 === 0) sentences.push(`Zoë had ${i % 34 === 0 ? 'blue' : 'brown'} eyes.`);
  if (i % 23 === 0) sentences.push('She saw saw a bird above the gate.');
  return { id: `scene-${i.toString().padStart(3, '0')}`, text: sentences.join(' ') };
});
process.stdout.write(JSON.stringify({ documentId: 'benchmark-novel-v2-xorshift32-20261027', knownCharacters: names, scenes }) + '\n');
