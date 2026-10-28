import fs from 'node:fs/promises';
import {Workbook, SpreadsheetFile} from '@oai/artifact-tool';
const dir = new URL('.', import.meta.url).pathname;
const data = JSON.parse(await fs.readFile(dir+'data.json','utf8'));
const wb = Workbook.create();
for (const [name,{columns,rows}] of Object.entries(data)) {
  const sh = wb.worksheets.add(name);
  sh.showGridLines = false;
  sh.getRange('A1').values = [[name === 'Businesses' ? 'Businesses with WFH flag 0 and combined score > 0.3' : 'Reviews for the selected businesses']];
  sh.getRange('A2').values = [[`${rows.length.toLocaleString()} records. Filter: wfh_flag = 0 and combined_wfh_score > 0.3.`]];
  sh.getRange('A3').values = [['Source: final_data_exploration.ipynb; jev_full_v1.csv; cached Yelp business dataset v4. Scores reconstructed from saved notebook.']];
  sh.getRangeByIndexes(4,0,1,columns.length).values=[columns];
  for(let i=0;i<rows.length;i+=500) sh.getRangeByIndexes(i+5,0,Math.min(500,rows.length-i),columns.length).values=rows.slice(i,i+500).map(row=>row.map(v=>typeof v==='string' && v.startsWith('=')?"'"+v:v));
  const all=sh.getRangeByIndexes(4,0,rows.length+1,columns.length);
  all.format.font.name='Arial';
  all.format.font.size=10;
  all.format.columnWidth=23;
  all.format.rowHeight=22;
  sh.getRangeByIndexes(4,0,1,columns.length).format.rowHeight=44;
  sh.getRangeByIndexes(4,0,1,columns.length).format.wrapText=true;
  sh.getRangeByIndexes(4,0,1,columns.length).format.fill='#243746';
  sh.getRangeByIndexes(4,0,1,columns.length).format.font.color='#FFFFFF';
  sh.getRangeByIndexes(4,0,1,columns.length).format.font.bold=true;
  sh.getRange('A:A').format.columnWidth=36;
  sh.getRange('B:B').format.columnWidth=29;
  sh.freezePanes.freezeRows(5);
  sh.tables.add(all.address ?? `A5:${String.fromCharCode(64+columns.length)}${rows.length+5}`,true,name+'Data');
  columns.forEach((col,j)=>{
    if(col.includes('score') || col==='stars') sh.getRangeByIndexes(5,j,rows.length,1).setNumberFormat('0.000');
    if(col==='text') {
      const r=sh.getRangeByIndexes(5,j,rows.length,1);
      r.format.columnWidth=120;
      r.format.wrapText=true;
      r.format.verticalAlignment='top';
      for(let i=0;i<rows.length;i++) sh.getRangeByIndexes(i+5,0,1,columns.length).format.rowHeight=Math.min(409,Math.max(40,(Math.ceil(String(rows[i][j]).length/130)+String(rows[i][j]).split('\n').length)*13));
    }
  });
  sh.getRange('A1').format.font.size=14;
  sh.getRange('A1').format.font.bold=true;
}
wb.recalculate();
console.log((await wb.inspect({kind:'table',range:'Businesses!A5:F8',tableMaxRows:4,tableMaxCols:6})).ndjson);
await (await SpreadsheetFile.exportXlsx(wb)).save(dir+'wfh_flag_0_score_above_0_3.xlsx');
for(const name of ['Businesses','Reviews']) {
 const preview=await wb.render({sheetName:name,range:name==='Businesses'?'A5:F10':'A5:D7',scale:1,format:'png'});
 await fs.writeFile(dir+name+'.png',new Uint8Array(await preview.arrayBuffer()));
}
console.log('Export complete');
