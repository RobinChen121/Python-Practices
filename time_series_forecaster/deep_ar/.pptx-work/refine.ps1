$ErrorActionPreference='Stop'
$a=New-Object -ComObject PowerPoint.Application
$p=$a.Presentations.Open((Join-Path (Get-Location) '.pptx-work\clean.pptx'),0,0,0)
$navy=77*65536+51*256
$accent=65*65536+0*256+170
$issues=@()
foreach($s in $p.Slides){
 $n=$s.SlideIndex
 foreach($sh in $s.Shapes){
  if($sh.HasTextFrame -ne -1 -or $sh.TextFrame.HasText -ne -1){continue}
  $tr=$sh.TextFrame.TextRange
  if($n -notin @(1,14,15,16,17,18,20) -and $sh.Top -lt 35 -and $tr.Text.Trim().Length -gt 3 -and $sh.Height -lt 48){
   $sh.Left=12; $sh.Width=324; $sh.Top=12
   $sh.TextFrame.MarginLeft=0;$sh.TextFrame.MarginRight=0;$sh.TextFrame.MarginTop=0;$sh.TextFrame.MarginBottom=0
   $tr.Font.Name='Arial';$tr.Font.Bold=-1;$tr.Font.Color.RGB=$navy
   $tr.Font.Size=17
   if($tr.Text.Length -gt 45){$tr.Font.Size=15}
   $sh.Height=[Math]::Max(25,$tr.BoundHeight+2)
  }
  if($sh.Top -gt 245 -and $tr.Text.Trim() -match '^\d+$'){$tr.Font.Name='Calibri';$tr.Font.Size=7;$tr.Font.Color.RGB=100*65536+100*256+100}
  if($n -notin @(1,14,15,16,17,18,20) -and $sh.Top -ge 35 -and $sh.Top -lt 245){
   $tr.ParagraphFormat.Alignment=1
   if($sh.Width -gt 280 -and $sh.Left -lt 20){$sh.Left=24;$sh.Width=312}
   if($sh.Left+$sh.Width -gt 340){$sh.Width=340-$sh.Left}
  }
 }
}
# Repair the crowded title and retain the original wording.
$t=$p.Slides.Item(25).Shapes.Item('object 38');$t.TextFrame.TextRange.Font.Size=14.5;$t.Width=324
$b=$p.Slides.Item(25).Shapes.Item('TextBox 1');$b.Width=222;$b.TextFrame.TextRange.Font.Size=16
# Improve the cover's contrast while keeping its university artwork.
$c=$p.Slides.Item(1).Shapes.Item('object 2');$c.TextFrame.TextRange.Font.Name='Arial';$c.TextFrame.TextRange.Font.Color.RGB=$navy
$c.TextFrame.TextRange.Paragraphs(1).Font.Size=14;$c.TextFrame.TextRange.Paragraphs(2).Font.Size=17
$c.TextFrame.TextRange.Paragraphs(2).Font.Bold=-1
# Give the journal list and its explanatory label separate areas.
$q=$p.Slides.Item(29).Shapes.Item('TextBox 3');$q.Top=86;$q.Height=153;$q.TextFrame.TextRange.Font.Size=10
$l=$p.Slides.Item(29).Shapes.Item('Rectangle: Rounded Corners 5');$l.Left=24;$l.Top=42;$l.Width=308;$l.Height=35
# Slightly enlarge dense case-study paragraphs only when they fit.
foreach($n in @(51,52,53)){
 foreach($sh in $p.Slides.Item($n).Shapes){
  if($sh.HasTextFrame -eq -1 -and $sh.TextFrame.HasText -eq -1 -and $sh.Top -gt 40 -and $sh.Top -lt 240){
   $sh.TextFrame.MarginLeft=3;$sh.TextFrame.MarginRight=3
   $sh.TextFrame.TextRange.ParagraphFormat.Alignment=1
  }
 }
}
# Fit text inside its existing box and record any remaining issues.
foreach($s in $p.Slides){foreach($sh in $s.Shapes){
 if($sh.HasTextFrame -eq -1 -and $sh.TextFrame.HasText -eq -1){
  $tf=$sh.TextFrame;$tr=$tf.TextRange
  if($s.SlideIndex -ne 35 -and $sh.Top -gt 35 -and $sh.Top -lt 245 -and $sh.Width -gt 100){
   $avail=245-$sh.Top
   if($tr.BoundHeight+$tf.MarginTop+$tf.MarginBottom -gt $sh.Height+2){$newHeight=[double]($tr.BoundHeight)+[double]($tf.MarginTop)+[double]($tf.MarginBottom)+2; if($newHeight -gt $avail){$newHeight=$avail}; $sh.Height=[single]$newHeight}
   if($tr.BoundHeight -gt $sh.Height+2){$issues += "Slide $($s.SlideIndex) $($sh.Name): text height $($tr.BoundHeight) box $($sh.Height)"}
  }
 }
}}
$final=Join-Path (Get-Location) 'output\Lecture 1 - Introduction - Refined.pptx'
$p.SaveAs($final,24)
$p.Export((Join-Path (Get-Location) '.pptx-work\after'),'PNG',1200,902)
$issues | Set-Content '.pptx-work/fit-issues.txt'
Write-Output "Saved $final";Write-Output $issues
$p.Close()


