from zipfile import ZipFile, ZIP_DEFLATED
from lxml import etree as E
from pathlib import Path
f=Path('output/Lecture 1 - Introduction - Refined.pptx');t=Path('.pptx-work/final-clean.pptx')
with ZipFile(f) as z,ZipFile(t,'w',ZIP_DEFLATED) as o:
 for n in z.namelist():
  if n.startswith('ppt/fonts/'):continue
  data=z.read(n)
  if n=='ppt/presentation.xml' or n.endswith('.rels') or n=='[Content_Types].xml':
   root=E.fromstring(data)
   for e in list(root.iter()):
    local=E.QName(e).localname
    if local=='embeddedFontLst' or (local=='Relationship' and ('/font' in e.get('Type',''))) or (local=='Override' and e.get('PartName','').startswith('/ppt/fonts/')) or (local=='Default' and e.get('Extension','') in ('fntdata','odttf')):
     e.getparent().remove(e)
   data=E.tostring(root,xml_declaration=True,encoding='UTF-8',standalone=True)
  o.writestr(n,data)
f.write_bytes(t.read_bytes())
print('Embedded fonts removed. Size:',f.stat().st_size)
