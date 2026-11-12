$ErrorActionPreference='Stop'
$a=New-Object -ComObject PowerPoint.Application
$f=Join-Path (Get-Location) 'output\Lecture 1 - Introduction - Refined.pptx'
$p=$a.Presentations.Open($f,0,0,0)
$sh=$p.Slides.Item(46).Shapes.Item('Rectangle 3');$sh.Left=24;$sh.Width=312;$sh.Top=43;$sh.Height=198;$sh.TextFrame.TextRange.Font.Size=12
foreach($n in @(23,47,50)) {foreach($sh in $p.Slides.Item($n).Shapes){if($sh.HasTextFrame -eq -1 -and $sh.TextFrame.HasText -eq -1 -and $sh.Type -eq 1){$sh.TextFrame.TextRange.ParagraphFormat.Alignment=2}}}
$p.Save();$p.Export((Join-Path (Get-Location) '.pptx-work\after'),'PNG',1200,902);$p.Close()
