const HexMap = (() => {
  const MAP_W = 30; // axial q range
  const MAP_H = 30; // axial r range
  const SQRT3 = Math.sqrt(3);

  function getCss(varName) {
    return getComputedStyle(document.documentElement).getPropertyValue(varName).trim();
  }

  function axialToCube(q, r) { return { x: q, z: r, y: -q - r }; }
  function cubeToAxial(x, y, z) { return { q: x, r: z }; }
  function cubeRound(x, y, z) {
    let rx = Math.round(x), ry = Math.round(y), rz = Math.round(z);
    const dx = Math.abs(rx - x), dy = Math.abs(ry - y), dz = Math.abs(rz - z);
    if (dx > dy && dx > dz) rx = -ry - rz;
    else if (dy > dz) ry = -rx - rz;
    else rz = -rx - ry;
    return { x: rx, y: ry, z: rz };
  }
  function axialRound(q, r) {
    const cr = cubeRound(q, -q - r, r);
    const ar = cubeToAxial(cr.x, cr.y, cr.z);
    return { q: ar.q, r: ar.r };
  }
  function offsetToAxial(col, row) {
    const q = col - ((row - (row & 1)) >> 1);
    const r = row;
    return { q, r };
  }
  function axialToOffset(q, r) {
    const col = q + ((r - (r & 1)) >> 1);
    const row = r;
    return { col, row };
  }
  function offsetNeighbors(c, r) {
    const odd = r & 1;
    const deltas = odd
      ? [[+1,0],[+1,-1],[0,-1],[-1,0],[0,+1],[+1,+1]]
      : [[+1,0],[0,-1],[-1,-1],[-1,0],[-1,+1],[0,+1]];
    return deltas.map(([dc, dr]) => ({ x: c + dc, y: r + dr }));
  }
  function hexDistance(a, b) {
    const aa = offsetToAxial(a.x, a.y);
    const bb = offsetToAxial(b.x, b.y);
    const ac = axialToCube(aa.q, aa.r);
    const bc = axialToCube(bb.q, bb.r);
    return Math.max(
      Math.abs(ac.x - bc.x),
      Math.abs(ac.y - bc.y),
      Math.abs(ac.z - bc.z)
    );
  }
  function nextStepOnHexLine(from, to) {
    const A = offsetToAxial(from.x, from.y);
    const B = offsetToAxial(to.x, to.y);
    const Ac = axialToCube(A.q, A.r);
    const Bc = axialToCube(B.q, B.r);
    const N = Math.max(
      Math.abs(Ac.x - Bc.x),
      Math.abs(Ac.y - Bc.y),
      Math.abs(Ac.z - Bc.z)
    );
    if (N <= 0) return null;
    const t = 1 / N;
    const nx = Ac.x + (Bc.x - Ac.x) * t;
    const ny = Ac.y + (Bc.y - Ac.y) * t;
    const nz = Ac.z + (Bc.z - Ac.z) * t;
    const cr = cubeRound(nx, ny, nz);
    const ax = cubeToAxial(cr.x, cr.y, cr.z);
    const off = axialToOffset(ax.q, ax.r);
    return { x: off.col, y: off.row };
  }
  function pointAtHexLineDistance(origin, target, dist) {
    let p = { x: origin.x, y: origin.y };
    for (let i = 0; i < dist; i++) {
      const n = nextStepOnHexLine(p, target);
      if (!n) break;
      p = n;
    }
    return p;
  }
  function clampTargetToRange(origin, target, maxRange) {
    const d = hexDistance(origin, target);
    if (d <= maxRange) return { x: target.x, y: target.y };
    return pointAtHexLineDistance(origin, target, maxRange);
  }

  function makeHexRenderer(canvas, W, H, getTileFn) {
    // local renderer state
    const SQ3 = Math.sqrt(3);
    let HEX = 10, ORX = 0, ORY = 0;
    const ctx = canvas.getContext('2d');
    function compute() {
      const sizeByW = canvas.width / (SQ3 * (W + 0.5));
      HEX = Math.max(5, Math.floor(sizeByW));
      const mapPixelW = SQ3 * HEX * (W + 0.5);
      const mapPixelH = 1.5 * HEX * (H - 1) + 2 * HEX;
      canvas.width = Math.ceil(mapPixelW);
      canvas.height = Math.ceil(mapPixelH);
      ORX = HEX; ORY = HEX;
    }
    compute();
    function offsetToPixel(col, row) {
      const x = HEX * (SQ3 * (col + 0.5 * (row & 1))) + ORX;
      const y = HEX * (1.5 * row) + ORY; return [x, y];
    }
    function hexPolygon(cx, cy, size) {
      const pts = []; for (let i=0;i<6;i++){ const ang = Math.PI/180*(60*i-30); pts.push([cx + size*Math.cos(ang), cy + size*Math.sin(ang)]);} return pts;
    }
    function renderBackground() {
      ctx.clearRect(0,0,canvas.width, canvas.height);
      ctx.fillStyle = getCss('--water'); ctx.fillRect(0,0,canvas.width, canvas.height);
      for (let r=0;r<H;r++){
        for (let c=0;c<W;c++){
          const [px,py] = offsetToPixel(c,r); const poly = hexPolygon(px,py,HEX);
          ctx.beginPath(); ctx.moveTo(poly[0][0], poly[0][1]); for (let i=1;i<poly.length;i++) ctx.lineTo(poly[i][0], poly[i][1]); ctx.closePath();
          ctx.fillStyle = (getTileFn(c,r)===1) ? getCss('--island') : getCss('--water'); ctx.fill();
          ctx.strokeStyle = getCss('--grid'); ctx.lineWidth = 1; ctx.stroke();
        }
      }
    }
    function renderVisibilityOverlay(visSet, color='rgba(255,255,255,0.14)') {
      if (!visSet || visSet.size === 0) return;
      for (let r = 0; r < H; r++) {
        for (let c = 0; c < W; c++) {
          if (!visSet.has(`${c},${r}`)) continue;
          const [px, py] = offsetToPixel(c, r);
          const poly = hexPolygon(px, py, HEX);
          ctx.beginPath();
          ctx.moveTo(poly[0][0], poly[0][1]);
          for (let i = 1; i < poly.length; i++) ctx.lineTo(poly[i][0], poly[i][1]);
          ctx.closePath();
          ctx.fillStyle = color;
          ctx.fill();
        }
      }
    }
    function drawHp(px,py,hp,hp_max,color){
      hp_max = (typeof hp_max === 'number') ? hp_max : 0;
      if( hp_max <= 0 ) return;
      hp = (typeof hp === 'number') ? hp : hp_max;
      const w=HEX*1.6,h=4;
      const x=Math.round(px - w/2), y=Math.round(py - h - 1);
      const ratio=Math.max(0,Math.min(1,hp/hp_max));
      ctx.fillStyle='rgba(0,0,0,0.6)'; ctx.fillRect(x,y,w,h);
      ctx.fillStyle=color; ctx.fillRect(x,y,Math.round(w*ratio),h);
    }
    function drawCarrier(x,y,color,hp,max){
      if(x==null||y==null) return;
      const [cx0,cy0]=offsetToPixel(x,y);
      const cx=cx0-HEX, cy=cy0-HEX;
      ctx.strokeStyle='rgba(0,0,0,0.85)';
      ctx.lineWidth=4; 
      ctx.strokeRect(cx+3,cy+3,HEX*2-6,HEX*2-6);
      ctx.fillStyle=color;
      ctx.fillRect(cx+4,cy+4,HEX*2-8,HEX*2-8);
      ctx.strokeStyle='rgba(255,255,255,0.35)';
      ctx.lineWidth=1.5;
      ctx.strokeRect(cx+4,cy+4,HEX*2-8,HEX*2-8);
      drawHp(cx0,cy+3,hp,max,color==='red'?'#ff9a9a':'#6ad4ff');
    }
    function drawCarrierStyled(x,y,color,{memory=false}={}){
      if(x==null||y==null) return;
      const [cx0,cy0]=offsetToPixel(x,y);
      const cx=cx0-HEX, cy=cy0-HEX;
      ctx.save();
      if (memory) ctx.globalAlpha = 0.55;
      ctx.strokeStyle='rgba(0,0,0,0.85)';
      ctx.lineWidth=4;
      ctx.strokeRect(cx+3,cy+3,HEX*2-6,HEX*2-6);
      ctx.fillStyle=color;
      ctx.fillRect(cx+4,cy+4,HEX*2-8,HEX*2-8);
      ctx.strokeStyle='rgba(255,255,255,0.35)';
      ctx.lineWidth=1.5;
      if (memory) ctx.setLineDash([4,3]);
      ctx.strokeRect(cx+4,cy+4,HEX*2-8,HEX*2-8);
      ctx.restore();
    }
    function drawSquadron(x,y,color,hp,max){
      if(x==null||y==null) return;
      const [px,py]=offsetToPixel(x,y);
      const r=Math.max(4, Math.round(HEX*0.6));
      ctx.beginPath(); ctx.arc(px, py, r+2, 0, Math.PI*2); ctx.strokeStyle='rgba(0,0,0,0.85)'; ctx.lineWidth=4; ctx.stroke();
      ctx.beginPath(); ctx.arc(px, py, r, 0, Math.PI*2); ctx.fillStyle=color; ctx.fill();
      ctx.strokeStyle='rgba(255,255,255,0.35)'; ctx.lineWidth=1.5; ctx.stroke();
      drawHp(px,Math.round(py-r-2),hp,max,color==='red'?'#ff9a9a':'#f2c14e');
    }
    function drawDiamond(x,y,color){
       if(x==null||y==null) return;
       const [px,py]=offsetToPixel(x,y);
       const r=Math.max(4, Math.round(HEX*0.6));
      const pts=[[px,py-r],[px+r,py],[px,py+r],[px-r,py]];
      ctx.beginPath(); ctx.moveTo(pts[0][0], pts[0][1]); for (let i=1;i<pts.length;i++) ctx.lineTo(pts[i][0], pts[i][1]); ctx.closePath();
      // halo
      ctx.strokeStyle='rgba(0,0,0,0.85)'; ctx.lineWidth=4; ctx.stroke();
      // fill
      ctx.fillStyle=color; ctx.fill();
      // border
      ctx.strokeStyle='rgba(255,255,255,0.35)'; ctx.lineWidth=1.5; ctx.stroke();
    }
    function drawDiamondStyled(x,y,color,{memory=false}={}){ if(x==null||y==null) return; const [px,py]=offsetToPixel(x,y); const r=Math.max(4, Math.round(HEX*0.6)); const pts=[[px,py-r],[px+r,py],[px,py+r],[px-r,py]]; ctx.save(); if (memory) ctx.globalAlpha = 0.55; ctx.beginPath(); ctx.moveTo(pts[0][0], pts[0][1]); for (let i=1;i<pts.length;i++) ctx.lineTo(pts[i][0], pts[i][1]); ctx.closePath(); ctx.strokeStyle='rgba(0,0,0,0.85)'; ctx.lineWidth=4; ctx.stroke(); ctx.fillStyle=color; ctx.fill(); ctx.strokeStyle='rgba(255,255,255,0.35)'; ctx.lineWidth=1.5; if (memory) ctx.setLineDash([4,3]); ctx.stroke(); ctx.restore(); }
    function drawLine(x1,y1,x2,y2,color){ ctx.strokeStyle=color; ctx.lineWidth=2; ctx.beginPath(); const [sx,sy]=offsetToPixel(x1,y1); const [tx,ty]=offsetToPixel(x2,y2); ctx.moveTo(sx,sy); ctx.lineTo(tx,ty); ctx.stroke(); }

    function drawHexOutline(c, r, color, width=2, dash=null) {
      if (c==null || r==null) return;
      const [px, py] = offsetToPixel(c, r);
      const poly = hexPolygon(px, py, HEX);
      ctx.save();
      if (Array.isArray(dash)) ctx.setLineDash(dash);
      ctx.beginPath();
      ctx.moveTo(poly[0][0], poly[0][1]);
      for (let i = 1; i < poly.length; i++) ctx.lineTo(poly[i][0], poly[i][1]);
      ctx.closePath();
      ctx.strokeStyle = color;
      ctx.lineWidth = width;
      ctx.stroke();
      ctx.restore();
    }
    function tileFromEvent(e){ const rect=canvas.getBoundingClientRect(); const scaleX = canvas.width/rect.width, scaleY = canvas.height/rect.height; const mx=(e.clientX-rect.left)*scaleX, my=(e.clientY-rect.top)*scaleY; const [qf, rf] = pixelToAxial(mx,my); const { q, r } = axialRound(qf, rf); const off = axialToOffset(q,r); const c=off.col, rr=off.row; if (c<0||rr<0||c>=W||rr>=H) return null; return {x:c,y:rr}; }
    function pixelToAxial(px,py){ const x=(px-ORX)/HEX, y=(py-ORY)/HEX; const q=(Math.sqrt(3)/3)*x - (1/3)*y; const r=(2/3)*y; return [q,r]; }
    function axialToCube(q,r){ return { x:q, z:r, y:-q-r }; }
    function cubeToAxial(x,y,z){ return { q:x, r:z }; }
    function cubeRound(x,y,z){ let rx=Math.round(x), ry=Math.round(y), rz=Math.round(z); const dx=Math.abs(rx-x), dy=Math.abs(ry-y), dz=Math.abs(rz-z); if (dx>dy && dx>dz) rx = -ry - rz; else if (dy>dz) ry = -rx - rz; else rz = -rx - ry; return { x:rx, y:ry, z:rz }; }
    function axialRound(q,r){ const cr=cubeRound(q, -q - r, r); const ar=cubeToAxial(cr.x, cr.y, cr.z); return { q: ar.q, r: ar.r}; }
    function axialToOffset(q,r){ const col = q + ((r - (r & 1)) >> 1); const row = r; return { col, row };
    }
    // Range outline (ring of distance=range around center)
    function drawRangeOutline(cx, cy, range, color) {
      const pts = [];
      const [pcx, pcy] = offsetToPixel(cx, cy);
      for (let y = 0; y < H; y++) {
        for (let x = 0; x < W; x++) {
          if (hexDistance({ x, y }, { x: cx, y: cy }) === range) {
            const [px, py] = offsetToPixel(x, y);
            const ang = Math.atan2(py - pcy, px - pcx);
            pts.push({ px, py, ang });
          }
        }
      }
      if (pts.length < 6) return; // not enough to draw a ring
      pts.sort((a, b) => a.ang - b.ang);
      ctx.save();
      ctx.strokeStyle = color;
      ctx.lineWidth = 1.5;
      ctx.setLineDash([6, 4]);
      ctx.beginPath();
      ctx.moveTo(pts[0].px, pts[0].py);
      for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i].px, pts[i].py);
      ctx.closePath();
      ctx.stroke();
      ctx.restore();
    }

    // Check if a tile is a valid target for the given mode
    function isValidTarget({ mode, x, y, carrier, squadrons }) {
      if (mode === 'launch') {
        // 航空部隊の目的地は「陸地も可」
        const arr = Array.isArray(squadrons) ? squadrons : [];
        const bases = arr.filter(b => b && b.state === 'onboard' && (b.hp ?? 0) > 0);
        if (bases.length === 0) return false;
        const maxFuel = Math.max(...bases.map(b => (typeof b.fuel === 'number') ? b.fuel : 0));
        if (typeof carrier?.x !== 'number' || typeof carrier?.y !== 'number') return false;
        const d = hexDistance({ x, y }, { x: carrier.x, y: carrier.y });
        return d <= maxFuel;
      } else if (mode === 'move') {
        // 空母の移動は海のみ
        return getTileFn(x, y) === 0; // move only on sea tiles
      }
      return true;
    }
    // Unified hover outline with validity coloring
    function renderHoverOutline({ mode, hover, carrier, squadrons }) {
      if (!hover || hover.x == null || hover.y == null) return;
      const valid = isValidTarget({ mode, x: hover.x, y: hover.y, carrier, squadrons });
      const color = valid ? '#ffffff' : '#ff5c5c';
      drawHexOutline(hover.x, hover.y, color, 2);

      const label = `(${hover.x},${hover.y})`;
      const [px, py] = offsetToPixel(hover.x, hover.y);
      const fontSize = Math.max(12, Math.round(HEX * 1.1));
      const padX = Math.max(5, Math.round(HEX * 0.45));
      const padY = Math.max(3, Math.round(HEX * 0.3));
      const fallbackFont = getCss('--font-sans') || 'sans-serif';

      ctx.save();
      ctx.font = `${fontSize}px ${fallbackFont}`;
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';

      const metrics = ctx.measureText(label);
      const boxW = Math.ceil(metrics.width) + padX * 2;
      const boxH = Math.ceil(fontSize) + padY * 2;

      let labelX = px;
      let labelY = py - HEX * 1.6;
      if (labelY - boxH / 2 < 0) labelY = py + HEX * 1.2;

      let boxX = Math.round(labelX - boxW / 2);
      let boxY = Math.round(labelY - boxH / 2);

      if (boxX < 0) {
        const diff = -boxX;
        labelX += diff;
        boxX = 0;
      }
      if (boxX + boxW > canvas.width) {
        const diff = boxX + boxW - canvas.width;
        labelX -= diff;
        boxX = canvas.width - boxW;
      }
      if (boxY < 0) {
        const diff = -boxY;
        labelY += diff;
        boxY = 0;
      }
      if (boxY + boxH > canvas.height) {
        const diff = boxY + boxH - canvas.height;
        labelY -= diff;
        boxY = canvas.height - boxH;
      }

      ctx.fillStyle = 'rgba(0,0,0,0.7)';
      ctx.fillRect(boxX, boxY, boxW, boxH);
      ctx.fillStyle = '#ffffff';
      ctx.fillText(label, labelX, labelY);
      ctx.restore();
    }
    return { canvas, W, H, renderBackground, renderVisibilityOverlay, drawCarrier, drawCarrierStyled, drawSquadron, drawDiamond, drawDiamondStyled, tileFromEvent, drawLine, drawHexOutline, drawRangeOutline, renderHoverOutline, isValidTarget, getTileFn };
  }

  return {
    MAP_W, MAP_H, SQRT3, getCss,
    axialToCube, cubeToAxial, cubeRound, axialRound,
    offsetToAxial, axialToOffset, offsetNeighbors,
    hexDistance, nextStepOnHexLine, pointAtHexLineDistance, clampTargetToRange,
    makeHexRenderer
  };
})();
