"""Independent unchanged copy of the frozen format/final input transform."""
from fractions import Fraction
import re
import numpy as np
TOKEN=re.compile(r'[a-z]+|\d[\d,]*(?:\.\d+)?(?:/\d+)?|[^\w\s]',re.I)

NUM=re.compile(r'[-+]?\d[\d,]*(?:\.\d+)?(?:/\d+)?')


def format_final(questions):
    x=np.zeros((len(questions),4,20),np.float32);clips=0
    for qi,q in enumerate(questions):
        for i,text in enumerate(q.texts):
            words=TOKEN.findall(text);lines=text.splitlines()
            f=[np.log1p(len(text)),np.log1p(len(words)),np.log1p(len(lines)),float('####' in text),
                float('\\boxed' in text),np.log1p(text.count('=')),np.log1p(len(NUM.findall(text))),
                sum(map(len,words))/max(1,len(words)),np.log1p(text.count('(')+text.count(')')),
                np.log1p(text.count('!')),np.log1p(len(re.findall(r'(?m)^\s*\d+[.)]',text))),
                sum(not line.strip() for line in lines)/max(1,len(lines))]
            value=q.answers[i]
            v=np.zeros(8,np.float32)
            if value is not None:
                a=Fraction(value);absolute=min(abs(a),Fraction(10**12));clips+=abs(a)>10**12
                v[:]=[1,float(a>0)-float(a<0),np.log1p(float(absolute)),float(a==0),
                    float(a.denominator==1),np.log1p(min(a.denominator,10**12)),
                    np.log1p(min(abs(a.numerator),10**12)),1/(1+float(absolute))]
            x[qi,i]=[*f,*v]
    return x,int(clips)

