const root=document.getElementById('analysis-result');
document.querySelectorAll('.seek-time').forEach(button=>button.addEventListener('click',()=>{const video=document.getElementById('source-video');video.currentTime=Number(button.dataset.seconds);video.play()}));
const eventRows=[...document.querySelectorAll('.timeline-event')];
const typeFilter=document.getElementById('event-type-filter');
const labelFilter=document.getElementById('event-label-filter');
const trackFilter=document.getElementById('event-track-filter');
const scoreFilter=document.getElementById('event-score-filter');
const timeFromFilter=document.getElementById('event-time-from');
const timeToFilter=document.getElementById('event-time-to');
const emptyTimeline=document.getElementById('timeline-empty');
function filterTimeline(){
  if(!eventRows.length)return;
  const type=typeFilter?.value||'',label=labelFilter?.value||'',track=trackFilter?.value||'',minimum=scoreFilter?.value===''?null:Number(scoreFilter?.value),from=timeFromFilter?.value===''?null:Number(timeFromFilter?.value),to=timeToFilter?.value===''?null:Number(timeToFilter?.value);
  let visible=0;
  eventRows.forEach(row=>{
    const score=row.dataset.score===''?null:Number(row.dataset.score);
    const time=Number(row.dataset.time);
    const show=(!type||row.dataset.type===type)&&(!label||row.dataset.label===label)&&(!track||row.dataset.track===track)&&(minimum===null||(score!==null&&score>=minimum))&&(from===null||time>=from)&&(to===null||time<=to);
    row.hidden=!show;if(show)visible++;
  });
  if(emptyTimeline)emptyTimeline.hidden=visible!==0;
}
[typeFilter,labelFilter,trackFilter,scoreFilter,timeFromFilter,timeToFilter].forEach(control=>control?.addEventListener('input',filterTimeline));
if(root){
  const notice=document.getElementById('processing-notice'),message=document.getElementById('processing-message'),bar=document.getElementById('analysis-progress'),badge=document.getElementById('status-badge'),faceLabel=document.getElementById('face-status-label'),poseLabel=document.getElementById('pose-status-label'),objectLabel=document.getElementById('object-status-label');
  let finished=false;
  const noticeKey=`analysis-notice:${root.dataset.videoId}`;
  let pendingNotice=null;
  try{pendingNotice=sessionStorage.getItem(noticeKey);sessionStorage.removeItem(noticeKey)}catch(_){}
  const storeNotice=text=>{try{sessionStorage.setItem(noticeKey,text)}catch(_){}};
  if(pendingNotice){message.textContent=pendingNotice;notice.hidden=false;setTimeout(()=>{notice.hidden=true},5000)}
  async function poll(){
    if(finished)return;
    try{
      const response=await fetch(root.dataset.statusUrl,{headers:{Accept:'application/json'}});
      if(!response.ok)throw new Error('Status unavailable');
      const data=await response.json();
      badge.textContent=data.status;badge.className=`badge ${data.status}`;bar.value=data.progress||0;
      if(faceLabel&&data.face_status)faceLabel.textContent=data.face_status.replaceAll('_',' ');
      if(poseLabel&&data.pose_status)poseLabel.textContent=data.pose_status.replaceAll('_',' ');
      if(objectLabel&&data.object_status)objectLabel.textContent=data.object_status.replaceAll('_',' ');
      if(data.status==='failed'||data.status==='interrupted'){
        message.textContent=data.error||'Processing did not complete.';storeNotice(`Analysis ${data.status}: ${message.textContent}`);notice.hidden=false;finished=true;setTimeout(()=>location.reload(),1800);
      }else if(data.status==='complete'){
        message.textContent='Analysis completed. Refreshing the saved results…';storeNotice('Analysis completed. Review the saved results and any candidate events.');notice.hidden=false;finished=true;setTimeout(()=>location.reload(),1000);
      }else{
        notice.hidden=false;message.textContent=data.status==='queued'?'Waiting for an analysis worker…':`Processing video… ${data.progress||0}%`;setTimeout(poll,1400);
      }
    }catch(_){message.textContent='Cannot retrieve processing status right now. Retrying…';setTimeout(poll,3000)}
  }
  if(['queued','processing'].includes(badge.textContent.trim()))setTimeout(poll,800);
}
