from zipfile import ZipFile
from lxml import etree as E
from collections import Counter
from pathlib import Path
ns={'a':'http://schemas.openxmlformats.org/drawingml/2006/main'}
f=Path('output/Lecture 1 - Introduction - Refined.pptx')
with ZipFile('.pptx-work/source.pptx') as a,ZipFile(f) as b:
 slides=[n for n in a.namelist() if n.startswith('ppt/slides/slide') and n.endswith('.xml')]
 diffs=[];fonts=Counter()
 for n in slides:
  x=E.fromstring(a.read(n));y=E.fromstring(b.read(n))
  if ''.join(x.xpath('//a:t/text()',namespaces=ns))!=''.join(y.xpath('//a:t/text()',namespaces=ns)):diffs.append(n)
 for n in b.namelist():
  if n.endswith('.xml') and n.startswith('ppt/'):
   x=E.fromstring(b.read(n))
   fonts.update(x.xpath('//@typeface'))
 print('Slides',len(slides),'Text changes',diffs,'Fonts',dict(fonts),'Bytes',f.stat().st_size)
 print('Media preserved:',all(a.read(n)==b.read(n) for n in a.namelist() if n.startswith('ppt/media/') and n in b.namelist()))
