import fs from 'node:fs/promises';
import {FileBlob,PresentationFile} from 'file:///C:/Users/Administrator/.cache/codex-runtimes/codex-primary-runtime/dependencies/node/node_modules/@oai/artifact-tool/dist/artifact_tool.mjs';
try {
const p=await PresentationFile.importPptx(await FileBlob.load('D:/chenzhen/python-practices/time_series_forecaster/deep_ar/output/Lecture 1 - Introduction - Refined.pptx'));
console.log(await p.inspect({kind:'slide',maxChars:1000}));
} catch(e){ console.log(String(e)); process.exitCode=1; }
