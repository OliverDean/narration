#!/usr/bin/env python3

import argparse, os, sys, json, re, math
import spacy
from typing import Dict, List
from fastcoref import spacy_component
from rapidfuzz.fuzz import ratio

MALE_SPEAKERS = ["p225","p226","p227","p228","p229","p230","p231","p232","p233","p234"]
FEMALE_SPEAKERS = ["p236","p237","p238","p239","p240","p241","p243","p244","p245","p246"]
DEFAULT_STOPNAMES = {"mom","dad","mother","father","grandma","grandfather","sir","ma'am","mr","mrs","lady","gentleman"}

def load_json(path):
    return json.load(open(path,"r",encoding="utf-8")) if os.path.isfile(path) else {}

def save_json(data,path):
    with open(path,"w",encoding="utf-8") as f: json.dump(data,f,indent=2)

def remove_blacklisted(voice_map, synonyms, user_bl=None):
    black = {b.lower() for b in DEFAULT_STOPNAMES}
    if user_bl and os.path.isfile(user_bl):
        for s in json.load(open(user_bl,"r",encoding="utf-8")): black.add(s.strip().lower())
    bad_keys = [k for k in voice_map if k.lower() in black]
    for bk in bad_keys:
        del voice_map[bk]; synonyms.pop(bk,None)
    for c,v in synonyms.items():
        synonyms[c] = [x for x in v if x.lower() not in black]

def clean_variant(txt):
    txt = re.sub(r"[.,!?;:\"]+$","",txt.strip())
    txt = re.sub(r"[’']s\b","",txt,flags=re.IGNORECASE)
    return txt.strip()

def unify_synonyms_inplace(syn_dict,voice_map,th=80):
    for c,v in syn_dict.items():
        merged=[]; 
        for cand in v:
            cc=clean_variant(cand); done=False
            for i,e in enumerate(merged):
                if ratio(cc.lower(),e.lower())>=th:
                    if len(cc)<len(e): merged[i]=cc
                    done=True;break
            if not done: merged.append(cc)
        syn_dict[c]=merged
    can_list=list(syn_dict.keys())
    changed=True
    while changed:
        changed=False; n=len(can_list); i=0
        while i<n-1:
            j=i+1; merged_any=False
            while j<n:
                a,b=can_list[i],can_list[j]
                if ratio(a.lower(),b.lower())>=th:
                    for sb in syn_dict[b]:
                        if sb not in syn_dict[a]: syn_dict[a].append(sb)
                    syn_dict.pop(b,None)
                    voice_map.pop(b,None)
                    can_list.remove(b)
                    n=len(can_list)
                    changed=True; merged_any=True; break
                else: j+=1
            if not merged_any: i+=1

def gender_from_coref(e_span,doc):
    male,fem=0,0; mset={"he","him","his"}; fset={"she","her","hers"}
    for t in e_span.sent:
        w=t.lower_
        if w in mset: male+=1
        if w in fset: fem+=1
    if male>fem and male>0: return "male"
    elif fem>male and fem>0: return "female"
    return None

def next_speaker(gend,midx,fidx):
    if gend=="male": s=MALE_SPEAKERS[midx[0]%len(MALE_SPEAKERS)]; midx[0]+=1; return s
    elif gend=="female": s=FEMALE_SPEAKERS[fidx[0]%len(FEMALE_SPEAKERS)]; fidx[0]+=1; return s
    else: s=MALE_SPEAKERS[midx[0]%len(MALE_SPEAKERS)]; midx[0]+=1; return s

def process_text(text_data, doc_nlp, doc_coref, voice_map, synonyms, male_idx, fem_idx, fth):
    doc_ner=doc_nlp(text_data)
    cset=set(voice_map.keys())
    newC=0; newS=0
    for ent in doc_ner.ents:
        if ent.label_=="PERSON":
            nm=clean_variant(ent.text)
            if len(nm)<2: continue
            if nm not in voice_map:
                g=gender_from_coref(ent,doc_ner)
                voice_map[nm]={"speaker_id":next_speaker(g,male_idx,fem_idx),"gender":g if g else"unknown","desc":"auto"}
                synonyms.setdefault(nm,[])
                cset.add(nm); newC+=1
            best_sc=0; best_c=None
            for c in cset:
                sc=ratio(nm.lower(),c.lower())
                if sc>best_sc: best_sc=sc; best_c=c
            if best_sc>=fth and best_c:
                if nm not in synonyms[best_c]:
                    synonyms[best_c].append(nm)
                    newS+=1
    return newC,newS

def pipeline(txtfile,voicejson,synjson,blackjson,coref_model,fth,chunk=0):
    if not os.path.isfile(txtfile): print("[ERR] text not found");return
    vmap=load_json(voicejson); syn=load_json(synjson)
    remove_blacklisted(vmap,syn,blackjson)
    for k in list(vmap.keys()):
        if k not in syn: syn[k]=[]
    full=open(txtfile,"r",encoding="utf-8").read()
    nlp=spacy.load("en_core_web_sm"); nlp.add_pipe("fastcoref",config={"model_architecture":coref_model,"device":"cpu"})
    nlp_ner=spacy.load("en_core_web_sm")
    if chunk<=0: chunk=len(full)
    steps=math.ceil(len(full)/chunk)
    mm=[0]; ff=[0]
    ctot=0; stot=0
    for i in range(steps):
        st=i*chunk; en=min(len(full),(i+1)*chunk)
        part=full[st:en]
        nc,ns=process_text(part,nlp_ner,nlp,vmap,syn,mm,ff,fth)
        ctot+=nc; stot+=ns
    unify_synonyms_inplace(syn,vmap,fth)
    remove_blacklisted(vmap,syn,blackjson)
    save_json(vmap,voicejson); save_json(syn,synjson)
    print(f"Done. NewChars={ctot}, NewSyn={stot}.")

def main():
    p=argparse.ArgumentParser()
    p.add_argument("text_file")
    p.add_argument("--voice_map",default="character_voices.json")
    p.add_argument("--synonyms",default="character_synonyms.json")
    p.add_argument("--blacklist",default="black_list.json")
    p.add_argument("--coref_model",default="FCoref")
    p.add_argument("--fuzzy_thresh",type=int,default=80)
    p.add_argument("--chunk_size",type=int,default=0)
    a=p.parse_args()
    pipeline(a.text_file,a.voice_map,a.synonyms,a.blacklist,a.coref_model,a.fuzzy_thresh,a.chunk_size)

if __name__=="__main__": main()
