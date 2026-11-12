from zipfile import ZipFile,ZIP_DEFLATED
from lxml import etree as E
from pathlib import Path
src=Path('.pptx-work/source.pptx');dst=Path('.pptx-work/clean.pptx')
ns={'a':'http://schemas.openxmlformats.org/drawingml/2006/main','p':'http://schemas.openxmlformats.org/presentationml/2006/main'}
with ZipFile(src) as z,ZipFile(dst,'w',ZIP_DEFLATED) as out:
 for name in z.namelist():
  data=z.read(name)
  if name.endswith('.xml') and name.startswith('ppt/'):
   root=E.fromstring(data)
   for el in root.xpath('//*[@typeface]'):
    font=el.get('typeface')
    if font=='Cambria Math':continue
    if E.QName(el).localname=='buFont':el.set('typeface','Arial')
    elif E.QName(el).localname=='ea':el.set('typeface','Microsoft YaHei')
    elif font.startswith('+'):continue
    else:el.set('typeface','Calibri')
   for el in root.findall('.//a:buChar',ns):el.set('char','•')
   for el in root.findall('.//p:embeddedFontLst',ns):el.getparent().remove(el)
   data=E.tostring(root,xml_declaration=True,encoding='UTF-8',standalone=True)
  out.writestr(name,data)
print(dst)
