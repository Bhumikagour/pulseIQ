/* =========================================================================
   PulseIQ v3 — blood-pressure visual kit
   Small, dependency-free SVG builders shared by every page:
     PulseArt.gauge(sbp, dbp, opts)  -> BP dial with the standard adult bands
     PulseArt.heart(opts)            -> stylised glowing heart with a pulse line
     PulseArt.spark(values, opts)    -> sparkline (optional second series)
     PulseArt.icon(name)             -> line icons: bp, heart, drop, signal, pulse,
                                        cuff, brain, sensor, shield
   Everything is drawn from the values passed in; nothing here makes up data.
   ========================================================================= */
(function () {
  var uid = 0;
  function id(p) { uid += 1; return p + uid; }

  // Standard adult bands, graded on systolic (same rule the backend uses).
  var BANDS = [
    { name: 'Low',        from: 70,  to: 90,  color: '#8aa8ff' },
    { name: 'Normal',     from: 90,  to: 120, color: '#3ce685' },
    { name: 'Borderline', from: 120, to: 130, color: '#ffd166' },
    { name: 'Elevated',   from: 130, to: 140, color: '#ff9f43' },
    { name: 'High',       from: 140, to: 180, color: '#ff3d7f' }
  ];
  function bandOf(sbp, dbp) {
    if (sbp >= 140 || dbp >= 90) return BANDS[4];
    if (sbp >= 130 || dbp >= 80) return BANDS[3];
    if (sbp >= 120) return BANDS[2];
    if (sbp < 90) return BANDS[0];
    return BANDS[1];
  }

  // ---- BP gauge -------------------------------------------------------
  function gauge(sbp, dbp, o) {
    o = o || {};
    var W = 240, H = 150, cx = 120, cy = 128, r = 96, lo = 70, hi = 180;
    function ang(v) { v = Math.max(lo, Math.min(hi, v)); return Math.PI * (1 - (v - lo) / (hi - lo)); }
    function pt(a, rr) { return [cx + rr * Math.cos(a), cy - rr * Math.sin(a)]; }
    function arc(a0, a1, rr) {
      var p0 = pt(a0, rr), p1 = pt(a1, rr);
      return 'M' + p0[0].toFixed(1) + ',' + p0[1].toFixed(1) + ' A' + rr + ',' + rr + ' 0 0 1 ' + p1[0].toFixed(1) + ',' + p1[1].toFixed(1);
    }
    var gid = id('gg'), segs = '';
    BANDS.forEach(function (b) {
      segs += '<path d="' + arc(ang(b.from) - 0.012, ang(b.to) + 0.012, r) + '" stroke="' + b.color +
        '" stroke-width="14" fill="none" stroke-linecap="butt" opacity="0.9"/>';
    });
    // tick labels at band edges
    var ticks = '';
    [90, 120, 140].forEach(function (v) {
      var a = ang(v), p0 = pt(a, r - 12), p1 = pt(a, r + 12), pl = pt(a, r + 22);
      ticks += '<line x1="' + p0[0].toFixed(1) + '" y1="' + p0[1].toFixed(1) + '" x2="' + p1[0].toFixed(1) + '" y2="' + p1[1].toFixed(1) +
        '" stroke="var(--bg-deep)" stroke-width="2.5"/>' +
        '<text x="' + pl[0].toFixed(1) + '" y="' + (pl[1] + 3).toFixed(1) + '" text-anchor="middle" font-size="9" font-weight="700" fill="var(--ink-500)" font-family="JetBrains Mono,monospace">' + v + '</text>';
    });
    var has = sbp != null && !isNaN(sbp);
    var needle = '';
    if (has) {
      var a = ang(sbp), tip = pt(a, r - 4), b1 = pt(a + Math.PI / 2, 5), b2 = pt(a - Math.PI / 2, 5);
      needle = '<g class="pa-needle"><path d="M' + tip[0].toFixed(1) + ',' + tip[1].toFixed(1) + ' L' + b1[0].toFixed(1) + ',' + b1[1].toFixed(1) +
        ' L' + b2[0].toFixed(1) + ',' + b2[1].toFixed(1) + ' Z" fill="var(--ink-0)" filter="url(#' + gid + ')"/>' +
        '<circle cx="' + cx + '" cy="' + cy + '" r="8" fill="var(--panel-solid)" stroke="var(--ink-0)" stroke-width="2.5"/></g>';
    }
    var band = has ? bandOf(sbp, dbp) : null;
    return '<svg class="pa-gauge" viewBox="0 0 ' + W + ' ' + H + '" role="img" aria-label="' +
      (has ? 'Blood pressure ' + Math.round(sbp) + ' over ' + Math.round(dbp) + ', ' + band.name : 'Blood pressure gauge') + '">' +
      '<defs><filter id="' + gid + '" x="-50%" y="-50%" width="200%" height="200%"><feGaussianBlur stdDeviation="2" result="b"/>' +
      '<feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter></defs>' +
      '<path d="' + arc(Math.PI, 0, r) + '" stroke="var(--wash-3)" stroke-width="22" fill="none"/>' +
      segs + ticks + needle +
      '</svg>' +
      (o.caption === false ? '' :
        '<div class="pa-gauge-read">' +
          (has ? '<div class="n"><b>' + Math.round(sbp) + '</b><span>/</span><b class="d">' + Math.round(dbp) + '</b><small>mmHg</small></div>' +
                 '<div class="band" style="color:' + band.color + ';background:' + band.color + '1f;">' + band.name + '</div>'
               : '<div class="n"><b>—</b></div>') +
        '</div>');
  }

  // ---- Heart illustration ---------------------------------------------
  function heart(o) {
    o = o || {};
    var g1 = id('hg'), g2 = id('hs'), g3 = id('hb'), f1 = id('hf'), clip = id('hc');
    // Stylised four-chamber heart silhouette with aortic arch and vessels.
    var body = 'M200 360 C 150 330, 92 280, 84 214 C 77 158, 110 118, 158 116 C 178 115, 194 124, 204 138 ' +
               'C 214 120, 236 106, 262 108 C 312 112, 338 158, 326 216 C 314 276, 254 330, 200 360 Z';
    var aorta = 'M186 132 C 184 96, 196 64, 228 56 C 262 48, 288 70, 286 100 C 285 116, 276 126, 268 132';
    var pulm = 'M214 140 C 222 110, 242 92, 270 90';
    var cava = 'M150 124 C 144 96, 146 70, 156 50';
    var vein = 'M300 150 C 318 136, 340 134, 356 142';
    var coronary = 'M198 150 C 176 196, 170 250, 196 330 M204 176 C 236 204, 262 236, 274 274 M162 170 C 140 200, 128 232, 130 262 ' +
                   'M232 206 C 250 214, 272 214, 290 202 M186 238 C 172 250, 152 256, 132 252';
    return '<svg class="pa-heart" viewBox="0 0 420 420" role="img" aria-label="Illustration of a heart with a pulse line">' +
      '<defs>' +
        '<radialGradient id="' + g1 + '" cx="42%" cy="40%" r="70%"><stop offset="0%" stop-color="#ff6b9a"/>' +
          '<stop offset="45%" stop-color="#e0245e"/><stop offset="100%" stop-color="#5a0b2e"/></radialGradient>' +
        '<linearGradient id="' + g2 + '" x1="0" y1="0" x2="1" y2="1"><stop offset="0%" stop-color="#ff8fb3"/>' +
          '<stop offset="100%" stop-color="#b3124b"/></linearGradient>' +
        '<radialGradient id="' + g3 + '" cx="50%" cy="50%" r="50%"><stop offset="0%" stop-color="#ff3d7f" stop-opacity="0.55"/>' +
          '<stop offset="55%" stop-color="#b45cf7" stop-opacity="0.18"/><stop offset="100%" stop-color="#22e5f0" stop-opacity="0"/></radialGradient>' +
        '<filter id="' + f1 + '" x="-30%" y="-30%" width="160%" height="160%"><feGaussianBlur stdDeviation="6" result="b"/>' +
          '<feMerge><feMergeNode in="b"/><feMergeNode in="SourceGraphic"/></feMerge></filter>' +
        '<clipPath id="' + clip + '"><path d="' + body + '"/></clipPath>' +
      '</defs>' +
      '<circle cx="210" cy="210" r="200" fill="url(#' + g3 + ')"/>' +
      '<circle class="pa-ring" cx="210" cy="215" r="170" fill="none" stroke="rgba(34,229,240,0.28)" stroke-width="1" stroke-dasharray="3 7"/>' +
      '<circle cx="210" cy="215" r="140" fill="none" stroke="rgba(255,61,127,0.18)" stroke-width="1"/>' +
      // vessels behind the body
      '<g fill="none" stroke="url(#' + g2 + ')" stroke-linecap="round">' +
        '<path d="' + aorta + '" stroke-width="26"/><path d="' + pulm + '" stroke-width="18"/>' +
        '<path d="' + cava + '" stroke-width="16"/><path d="' + vein + '" stroke-width="12"/>' +
        '<path d="M232 60 C 230 42, 236 30, 246 22" stroke-width="9"/><path d="M256 56 C 260 40, 270 30, 282 26" stroke-width="9"/>' +
      '</g>' +
      '<path class="pa-heart-body" d="' + body + '" fill="url(#' + g1 + ')" filter="url(#' + f1 + ')"/>' +
      '<g clip-path="url(#' + clip + ')" fill="none" stroke="#ffb3c9" stroke-opacity="0.45" stroke-width="2" stroke-linecap="round">' +
        '<path d="' + coronary + '"/>' +
        '<ellipse cx="160" cy="180" rx="46" ry="30" fill="#ffffff" fill-opacity="0.08" stroke="none"/>' +
      '</g>' +
      '<path d="' + body + '" fill="none" stroke="#ff9cbc" stroke-opacity="0.7" stroke-width="1.5"/>' +
      // pulse line sweeping across
      '<path class="pa-pulse" d="M14 250 L120 250 L138 250 L150 214 L166 292 L184 186 L200 262 L214 244 L230 250 L406 250" fill="none" ' +
        'stroke="#7bf5fb" stroke-width="3" stroke-linecap="round" stroke-linejoin="round" filter="url(#' + f1 + ')"/>' +
      // HUD corner brackets
      '<g fill="none" stroke="rgba(123,245,251,0.55)" stroke-width="2">' +
        '<path d="M60 70 L60 50 L80 50"/><path d="M360 50 L380 50 L380 70"/><path d="M60 350 L60 370 L80 370"/><path d="M360 370 L380 370 L380 350"/>' +
      '</g>' +
      '</svg>';
  }

  // ---- Sparkline ------------------------------------------------------
  function spark(values, o) {
    o = o || {};
    var W = o.w || 120, H = o.h || 36, pad = 3;
    var series = [values].concat(o.second ? [o.second] : []);
    var all = [].concat.apply([], series).filter(function (v) { return v != null && !isNaN(v); });
    if (all.length < 2) return '';
    var lo = Math.min.apply(null, all), hi = Math.max.apply(null, all), span = (hi - lo) || 1;
    var colors = [o.color || 'var(--cyan)', o.color2 || 'var(--dbp)'], out = '', gid = id('sp');
    series.forEach(function (vals, k) {
      var n = vals.length, d = '';
      vals.forEach(function (v, i) {
        var x = pad + (i / (n - 1)) * (W - 2 * pad), y = pad + (1 - (v - lo) / span) * (H - 2 * pad);
        d += (i ? 'L' : 'M') + x.toFixed(1) + ',' + y.toFixed(1);
      });
      if (k === 0 && o.fill !== false)
        out += '<path d="' + d + 'L' + (W - pad) + ',' + H + 'L' + pad + ',' + H + 'Z" fill="url(#' + gid + ')"/>';
      out += '<path d="' + d + '" fill="none" stroke="' + colors[k] + '" stroke-width="' + (o.sw || 1.8) +
        '" stroke-linecap="round" stroke-linejoin="round" vector-effect="non-scaling-stroke"/>';
    });
    return '<svg class="pa-spark" viewBox="0 0 ' + W + ' ' + H + '" preserveAspectRatio="none" aria-hidden="true">' +
      '<defs><linearGradient id="' + gid + '" x1="0" y1="0" x2="0" y2="1"><stop offset="0%" stop-color="' + colors[0] +
      '" stop-opacity="0.28"/><stop offset="100%" stop-color="' + colors[0] + '" stop-opacity="0"/></linearGradient></defs>' + out + '</svg>';
  }

  // ---- Icons ----------------------------------------------------------
  var ICONS = {
    bp: '<path d="M4 15a8 8 0 1 1 16 0"/><path d="M12 15l4-5"/><circle cx="12" cy="15" r="1.4" fill="currentColor"/><path d="M6.5 19h11"/>',
    heart: '<path d="M12 20s-7-4.4-7-10a4 4 0 0 1 7-2.6A4 4 0 0 1 19 10c0 5.6-7 10-7 10z"/><path d="M3 12h4l1.5-2.5L11 15l1.5-3H21"/>',
    drop: '<path d="M12 3s6 6.6 6 11a6 6 0 0 1-12 0c0-4.4 6-11 6-11z"/><path d="M9 14.5a3 3 0 0 0 3 3"/>',
    signal: '<path d="M2 12h3l2-5 3 10 3-13 3 11 2-3h4"/>',
    pulse: '<path d="M3 12h4l2-6 4 12 2-6h6"/>',
    cuff: '<rect x="3" y="8" width="11" height="8" rx="2"/><path d="M14 11h3a3 3 0 0 1 3 3v2"/><circle cx="20" cy="18" r="1.6"/><path d="M6 8V6M11 8V6"/>',
    brain: '<path d="M9 4a3 3 0 0 0-3 3 3 3 0 0 0-2 5 3 3 0 0 0 2 5 3 3 0 0 0 3 3h1V4z"/><path d="M15 4a3 3 0 0 1 3 3 3 3 0 0 1 2 5 3 3 0 0 1-2 5 3 3 0 0 1-3 3h-1V4z"/>',
    sensor: '<path d="M8 21v-6a4 4 0 0 1 8 0v6"/><path d="M12 3v3M5.6 5.6l2.1 2.1M18.4 5.6l-2.1 2.1"/><circle cx="12" cy="15" r="1.5" fill="currentColor"/>',
    shield: '<path d="M12 3l7 3v6c0 4.5-3 7.8-7 9-4-1.2-7-4.5-7-9V6z"/><path d="M9 12l2 2 4-4"/>',
    pin: '<path d="M12 21s7-6.2 7-12a7 7 0 0 0-14 0c0 5.8 7 12 7 12z"/><circle cx="12" cy="9" r="2.5"/>',
    file: '<path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8z"/><path d="M14 3v5h5M9 13h6M9 17h4"/>',
    pill: '<rect x="3" y="9" width="18" height="7" rx="3.5" transform="rotate(-35 12 12.5)"/><path d="M9.5 8.5l5 7"/>',
    chat: '<path d="M4 5h16v11H9l-5 4z"/><path d="M8 10h8M8 13h5"/>',
    monitor: '<rect x="3" y="4" width="18" height="12" rx="2"/><path d="M6 11h3l1.5-3 2.5 6 1.5-3H18M9 20h6M12 16v4"/>',
    arrow: '<path d="M5 12h14M13 6l6 6-6 6"/>'
  };
  function icon(name, size) {
    return '<svg class="pa-ic" width="' + (size || 20) + '" height="' + (size || 20) + '" viewBox="0 0 24 24" fill="none" stroke="currentColor" ' +
      'stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">' + (ICONS[name] || '') + '</svg>';
  }

  // Fill any element with data-pa-icon="name" automatically.
  function hydrate(root) {
    (root || document).querySelectorAll('[data-pa-icon]').forEach(function (el) {
      if (!el.__pa) { el.innerHTML = icon(el.getAttribute('data-pa-icon'), +el.getAttribute('data-size') || 20); el.__pa = 1; }
    });
    // v3: artwork from the PulseIQ design reference (shared/img). The SVG heart()
    // above stays available as a fallback if the image fails to load.
    [['data-pa-heart', 'heart', 'Glowing illustration of a human heart'],
     ['data-pa-hand', 'hand', 'Illustration of a wrist with a glowing pulse sensor']].forEach(function (a) {
      (root || document).querySelectorAll('[' + a[0] + ']').forEach(function (el) {
        if (el.__pa) return;
        el.__pa = 1;
        var img = new Image();
        img.className = 'pa-art pa-art-' + a[1];
        img.alt = a[2];
        img.decoding = 'async';
        img.src = 'shared/img/' + a[1] + '-hd.webp';
        if (a[1] === 'heart') img.onerror = function () { el.innerHTML = heart(); };
        el.innerHTML = '';
        el.appendChild(img);
      });
    });
  }
  if (document.readyState === 'loading') document.addEventListener('DOMContentLoaded', function () { hydrate(); });
  else hydrate();

  window.PulseArt = { gauge: gauge, heart: heart, spark: spark, icon: icon, bandOf: bandOf, BANDS: BANDS, hydrate: hydrate };
})();
