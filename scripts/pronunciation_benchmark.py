"""Prepare and score an independent,pinned pronunciation reference;no production downloads."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import re

from fetch_literary_benchmark import check
from literary_benchmark import counts, fingerprint, write

ROOT=Path(__file__).resolve().parents[1]


def prepare(folder):
    spec=json.loads((ROOT/'docs/benchmarks/pronunciation-reference.json').read_text())
    cache=ROOT/spec['cache']
    for entry in spec['files']:
        path=cache/entry['path']
        if path.stat().st_size!=entry['size']:
            raise ValueError('pronunciation source size changed')
        check(path.read_bytes(),entry)
    phones=dict(line.split() for line in (cache/'cmudict.phones').read_text().splitlines() if line.strip())
    pronunciations=defaultdict(set)
    excluded=Counter()
    for line in (cache/'cmudict.dict').read_text().splitlines():
        fields=line.split('#',1)[0].split()
        if not fields:
            continue
        match=re.fullmatch(r'([a-z]{1,128})(?:\([0-9]+\))?',fields[0])
        if not match:
            excluded['non_ascii_alphabetic_entries']+=1
            continue
        symbols=[re.sub('[012]$','',phone) for phone in fields[1:]]
        if not symbols or any(phone not in phones for phone in symbols):
            raise ValueError('reference contains an unknown phoneme')
        syllables=sum(phones[phone]=='vowel' for phone in symbols)
        if syllables==0:
            excluded['zero_vowel_pronunciations']+=1
            continue
        pronunciations[match[1]].add(syllables)
    groups={}
    for split in ['development','final_holdout']:
        rows=[]
        prefixes=set()
        for word,values in sorted(pronunciations.items()):
            prefix=word[:4]
            heldout=int(hashlib.sha256((spec['split']['seed']+'\n'+prefix).encode()).hexdigest()[:8],16)%spec['split']['holdout_modulo']==0
            if heldout!=(split=='final_holdout'):
                continue
            rows.append({'word':word,'syllables':sorted(values)})
            prefixes.add(prefix)
        groups[split]=prefixes
        target=folder/split
        write(target/'inputs.json',[row['word'] for row in rows])
        write(target/'gold.json',rows)
        manifest={'protocolVersion':1,'source':spec,'split':split,'words':len(rows),'prefixGroups':len(prefixes),'excludedPronunciations':dict(excluded),
                  'inputsSha256':fingerprint((target/'inputs.json').read_bytes()),'goldSha256':fingerprint((target/'gold.json').read_bytes()),
                  'scope':'Lowercase ASCII alphabetic lexical items. Vowel-phone counts from published CMU pronunciations;any supplied variant accepted. Four-letter-prefix-disjoint splits,not guaranteed linguistic-family independence. No speech,numeral expansion,non-ASCII or reading-comprehension accuracy inferred.'}
        write(target/'manifest.json',manifest)
    if groups['development']&groups['final_holdout']:
        raise ValueError('prefix overlap between pronunciation splits')
    print(json.dumps({'developmentPrefixGroups':len(groups['development']),'holdoutPrefixGroups':len(groups['final_holdout']),'sourceRevision':spec['revision']}))


def score(folder,predictions,split,output):
    folder=folder/split
    manifest=json.loads((folder/'manifest.json').read_text())
    for name in ['inputs','gold']:
        if fingerprint((folder/(name+'.json')).read_bytes())!=manifest[name+'Sha256']:
            raise ValueError('pronunciation preparation changed')
    gold={row['word']:row['syllables'] for row in json.loads((folder/'gold.json').read_text())}
    rows=json.loads(predictions.read_text())
    actual=dict(rows)
    if len(actual)!=len(rows) or set(actual)!=set(gold) or any(type(value)is not int or value<0 for value in actual.values()):
        raise ValueError('pronunciation predictions are invalid,missing,or duplicate')
    correct=0
    absolute_error=0
    ambiguous=0
    poly=Counter()
    errors=[]
    for word,expected in gold.items():
        predicted=actual[word]
        compatible=predicted in expected
        correct+=compatible
        error=min(abs(predicted-value) for value in expected)
        absolute_error+=error
        labels={value>=3 for value in expected}
        if len(labels)==1:
            positive=next(iter(labels))
            poly.update(tp=int(positive and predicted>=3),fp=int(not positive and predicted>=3),fn=int(positive and predicted<3),tn=int(not positive and predicted<3))
        else:
            ambiguous+=1
        if not compatible:
            errors.append({'word':word,'expectedSyllables':expected,'predicted':predicted,'minimumAbsoluteError':error})
    total=len(gold)
    report={'dataset':manifest,'split':split,'predictionSha256':fingerprint(predictions.read_bytes()),'wordCount':total,
            'compatibleCount':correct,'variantCompatibleAccuracy':correct/total if total else None,'meanMinimumAbsoluteError':absolute_error/total if total else None,
            'polysyllabic':{**counts(poly['tp'],poly['fp'],poly['fn']),'trueNegatives':poly['tn'],'ambiguousReferenceWords':ambiguous,'assessmentCoverage':(total-ambiguous)/total if total else None},
            'limits':['Pronunciation-reference compatibility is not proof of perfect pronunciation,utterance-specific syllables or human comprehension.','Words have equal weight;this does not represent frequency-weighted manuscript readability accuracy.','Any listed pronunciation count is accepted;polysyllabic labels straddling2/3 are explicitly unassessed.'],
            'largestDevelopmentErrors':sorted(errors,key=lambda row:(-row['minimumAbsoluteError'],row['word']))[:30] if split=='development' else []}
    write(output,report)
    print(json.dumps({key:report[key] for key in ['split','wordCount','variantCompatibleAccuracy','meanMinimumAbsoluteError','polysyllabic']},indent=2))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode',choices=['prepare','score'])
    parser.add_argument('--folder',type=Path,default=ROOT/'target/external-validation/pronunciation-prepared')
    parser.add_argument('--predictions',type=Path)
    parser.add_argument('--split',choices=['development','final_holdout'],default='development')
    parser.add_argument('--finalize',action='store_true')
    parser.add_argument('--output',type=Path,default=ROOT/'docs/audit-2026-10-27/pronunciation-development.json')
    args=parser.parse_args()
    if args.mode=='prepare':
        prepare(args.folder)
    else:
        if args.split=='final_holdout' and not args.finalize:
            raise ValueError('final pronunciation scoring requires candidate freeze and --finalize')
        if args.predictions is None:
            raise ValueError('pronunciation predictions required')
        score(args.folder,args.predictions,args.split,args.output)


if __name__=='__main__':
    main()
