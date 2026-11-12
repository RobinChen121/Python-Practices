from PIL import Image,ImageOps,ImageDraw
from pathlib import Path
p=Path('.pptx-work/before')
files=sorted(p.glob('*.PNG'),key=lambda f:int(''.join(filter(str.isdigit,f.stem))))
for b in range(0,len(files),12):
 out=Image.new('RGB',(1280, (min(12,len(files)-b)+3)//4*265),'#dddddd')
 for j,f in enumerate(files[b:b+12]):
  im=Image.open(f).resize((312,234)); x=(j%4)*320;y=(j//4)*265
  out.paste(im,(x,y+22));ImageDraw.Draw(out).text((x+5,y+3),str(b+j+1),fill='black')
 out.save(f'.pptx-work/before-{b//12}.jpg')
