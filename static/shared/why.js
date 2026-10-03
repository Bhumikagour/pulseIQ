/* =========================================================================
   PulseIQ v3 — "Why this reading" panel
   One reusable panel shown next to every BP estimate (patient dashboard,
   ICU monitor, doctor roster). It explains the reading in three layers:

   1. Band        — which standard adult band the estimate falls in, and why
                    (the same rule the backend's classify_bp uses).
   2. Deep model  — Integrated Gradients on PPGResNetBiLSTM, the model that
                    produced the headline number: which input channel and which
                    part of each heartbeat the model relied on. Fast (~0.3 s).
   3. Classical   — SHAP on the classical model (AdaBoost/SVR), grouped into
                    physiological themes. Slower (~15 s the first time, cached
                    after), so pages can load it automatically or on a click.

   Nothing here is invented: every number comes from the backend for the
   exact window on screen.

   Usage:
     PulseWhy.mount(el, { key: 'rohit-sharma', window: 12, sbp: 117, dbp: 54,
                          shap: 'auto' | 'click', layout: 'row' | 'stack' });
   ========================================================================= */
(function () {
  var API = function () { return window.PULSEIQ_API_BASE || (location.port === '8090' ? 'http://localhost:8001' : location.origin); };

  var CSS = '' +
    '.why{background:var(--glass);border:1px solid var(--glass-border);border-radius:16px;padding:16px 18px;}' +
    '.why-head{display:flex;align-items:center;gap:10px;margin-bottom:12px;flex-wrap:wrap;}' +
    '.why-head h3{margin:0;font-size:13px;font-weight:800;color:var(--ink-0);}' +
    '.why-head .why-tag{font-size:9px;font-weight:800;letter-spacing:.06em;text-transform:uppercase;padding:3px 9px;border-radius:99px;' +
      'background:rgba(34,229,240,.1);color:var(--cyan-bright);border:1px solid rgba(34,229,240,.3);}' +
    '.why-head .grow{flex:1;}' +
    '.why-head a{font-size:11px;font-weight:700;color:var(--cyan-bright);text-decoration:none;}' +
    '.why-grid{display:grid;gap:14px;}' +
    '.why.row .why-grid{grid-template-columns:0.9fr 1.2fr 1.3fr;}' +
    '.why.row.noread .why-grid{grid-template-columns:1fr 1fr;}' +
    '.why-sec{background:var(--wash-1);border:1px solid var(--glass-border);border-radius:12px;padding:12px 14px;min-width:0;}' +
    '.why-sec .k{font-size:9.5px;font-weight:800;letter-spacing:.06em;text-transform:uppercase;color:var(--ink-500);margin-bottom:8px;}' +
    '.why-sec .k small{text-transform:none;letter-spacing:0;font-weight:600;color:var(--ink-600);}' +
    '.why-band{font-size:20px;font-weight:800;margin-bottom:4px;}' +
    '.why-band.normal{color:var(--green);} .why-band.elevated{color:var(--amber);} .why-band.high{color:var(--red);}' +
    '.why-txt{font-size:11.5px;line-height:1.55;color:var(--ink-300);}' +
    '.why-txt b{color:var(--ink-0);}' +
    '.why-muted{font-size:10.5px;line-height:1.5;color:var(--ink-600);margin-top:8px;}' +
    '.why-spark{width:100%;height:64px;display:block;margin:2px 0 8px;}' +
    '.why-beat{width:100%;height:auto;max-height:230px;display:block;margin:2px 0 10px;border-radius:10px;overflow:hidden;}' +
    '.why-phases{display:grid;grid-template-columns:repeat(3,1fr);gap:8px;margin:0 0 10px;}' +
    '.why-phase{background:var(--wash-1);border:1px solid var(--glass-border);border-radius:10px;padding:8px 10px;}' +
    '.why-phase.hi{border-color:var(--cyan-bright);box-shadow:0 0 0 1px var(--cyan-bright) inset;}' +
    '.why-phase .n{font-size:10px;font-weight:700;color:var(--ink-300);}' +
    '.why-phase .v{font-family:"JetBrains Mono",monospace;font-size:17px;font-weight:800;color:var(--ink-0);margin-top:2px;}' +
    '.why-phase .s{font-size:9.5px;color:var(--ink-600);}' +
    '.why-k2{font-size:9px;letter-spacing:0.07em;text-transform:uppercase;color:var(--ink-600);font-weight:700;margin-top:4px;}' +
    '.why-ch{display:flex;height:9px;border-radius:99px;overflow:hidden;margin:6px 0 5px;background:var(--wash-2);}' +
    '.why-ch span{display:block;height:100%;}' +
    '.why-leg{display:flex;flex-wrap:wrap;gap:4px 12px;font-size:10.5px;color:var(--ink-400);}' +
    '.why-leg i{display:inline-block;width:8px;height:8px;border-radius:2px;margin-right:5px;vertical-align:0;}' +
    '.why-f{display:grid;grid-template-columns:1fr auto;gap:2px 10px;align-items:center;margin-bottom:8px;}' +
    '.why-f .n{font-size:11.5px;font-weight:700;color:var(--ink-100);min-width:0;}' +
    '.why-f .v{font-family:"JetBrains Mono",monospace;font-size:11.5px;font-weight:800;}' +
    '.why-f .v.up{color:var(--red);} .why-f .v.down{color:var(--green);}' +
    '.why-f .bar{grid-column:1/-1;height:5px;border-radius:99px;background:var(--wash-2);overflow:hidden;}' +
    '.why-f .bar span{display:block;height:100%;border-radius:99px;}' +
    '.why-btn{background:var(--glass);border:1px solid var(--glass-border-2);color:var(--ink-0);padding:8px 12px;border-radius:9px;' +
      'font-size:11px;font-weight:700;cursor:pointer;font-family:inherit;}' +
    '.why-btn:hover{border-color:rgba(34,229,240,.5);}' +
    '.why-spin{display:inline-block;width:11px;height:11px;border:2px solid var(--ink-600);border-top-color:var(--cyan-bright);' +
      'border-radius:50%;animation:whyspin .8s linear infinite;vertical-align:-2px;margin-right:7px;}' +
    '@keyframes whyspin{to{transform:rotate(360deg);}}' +
    '@media(max-width:900px){.why.row .why-grid{grid-template-columns:1fr;}}';

  function injectCss() {
    if (document.getElementById('why-css')) return;
    var s = document.createElement('style');
    s.id = 'why-css';
    s.textContent = CSS;
    document.head.appendChild(s);
  }

  function esc(s) {
    return String(s).replace(/[&<>"']/g, function (c) {
      return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c];
    });
  }

  // Same rule as the backend's classify_bp (graded on systolic; diastolic only escalates).
  function band(sbp, dbp) {
    if (sbp >= 140 || dbp >= 90) return { cls: 'high', label: 'High',
      why: 'systolic <b>' + sbp + '</b> is at or above 140, or diastolic <b>' + dbp + '</b> is at or above 90' };
    if (sbp >= 130 || dbp >= 80) return { cls: 'elevated', label: 'Elevated',
      why: 'systolic <b>' + sbp + '</b> is at or above 130, or diastolic <b>' + dbp + '</b> is at or above 80' };
    if (sbp >= 120) return { cls: 'elevated', label: 'Borderline',
      why: 'systolic <b>' + sbp + '</b> is in the 120–129 range while diastolic stays under 80' };
    if (sbp < 90) return { cls: 'elevated', label: 'Low', why: 'systolic <b>' + sbp + '</b> is below 90' };
    return { cls: 'normal', label: 'Normal',
      why: 'systolic <b>' + sbp + '</b> is under 120 and diastolic <b>' + dbp + '</b> is under 80' };
  }

  var CH = [
    { k: 'ppg', name: 'Pulse shape', tech: 'PPG', color: 'var(--spo2)' },
    { k: 'vpg', name: 'Rise speed', tech: 'VPG', color: 'var(--hr)' },
    { k: 'apg', name: 'Rise acceleration', tech: 'APG', color: 'var(--sbp)' }
  ];

  // ---- Integrated Gradients summary ------------------------------------
  function peaks(z, minDist) {
    var out = [], last = -1e9;
    for (var i = 1; i < z.length - 1; i++) {
      if (z[i] > z[i - 1] && z[i] >= z[i + 1] && z[i] > 0.2 && i - last >= minDist) { out.push(i); last = i; }
    }
    return out;
  }

  // Label every sample by where it sits in its heartbeat:
  // foot -> peak = upstroke; first 40% after the peak = peak & dicrotic notch;
  // the rest until the next foot = diastolic decay.
  function beatPhases(ppg, fs) {
    var n = ppg.length, m = 0, sd = 0, i;
    for (i = 0; i < n; i++) m += ppg[i];
    m /= n;
    for (i = 0; i < n; i++) sd += (ppg[i] - m) * (ppg[i] - m);
    sd = Math.sqrt(sd / n) || 1;
    var z = ppg.map(function (v) { return (v - m) / sd; });
    var pk = peaks(z, Math.max(3, Math.round(0.33 * fs)));
    var lab = new Array(n).fill(null);
    if (pk.length < 3) return lab;
    var feet = [];
    for (var j = 0; j < pk.length; j++) {
      var lo = j === 0 ? Math.max(0, pk[0] - Math.round(0.5 * fs)) : pk[j - 1];
      var f = lo;
      for (i = lo; i < pk[j]; i++) if (z[i] < z[f]) f = i;
      feet.push(f);
    }
    for (j = 0; j < pk.length; j++) {
      for (i = feet[j]; i < pk[j]; i++) lab[i] = 'upstroke';
      if (j + 1 < pk.length) {
        var nextFoot = feet[j + 1], cut = pk[j] + Math.round((nextFoot - pk[j]) * 0.4);
        for (i = pk[j]; i < cut; i++) lab[i] = 'notch';
        for (i = cut; i < nextFoot; i++) lab[i] = 'decay';
      }
    }
    return lab;
  }

  // Plain words for patients; clinical terms for doctors.
  var PHASE_NAME = { upstroke: 'rise of each heartbeat', notch: 'peak of each heartbeat and the small dip after it',
                     decay: 'slow fall between heartbeats' };
  var PHASE_TECH = { upstroke: 'systolic upstroke', notch: 'peak and dicrotic notch', decay: 'diastolic decay' };

  // v3: one "average heartbeat". Each beat's attention is scaled to 100% before
  // averaging, so the model's habit of weighting the end of the window more
  // heavily doesn't masquerade as a finding. Falls back to igHtmlWindow().
  var BEAT_LAB = { upstroke: ['Rise', 'Systolic upstroke'], notch: ['Peak & dip', 'Peak & dicrotic notch'],
                   decay: ['Slow fall', 'Diastolic decay'] };
  function igHtml(d, tech) {
    var b = d.beat;
    if (b === undefined) {
      // Page is newer than the server: an old backend without the beat view is running.
      return '<div class="why-txt"><b>The PulseIQ server is running an older version.</b> Stop it (Ctrl+C) and start it again ' +
        'with <code>python -m uvicorn main:app --port 8001</code>, then reload this page.</div>';
    }
    if (!b || !b.attention || b.attention.length < 10) {
      // Beats couldn't be separated cleanly. Don't fall back to the 12-second
      // strip: its end-of-window bias would mislead. Show the signal split only.
      return '<div class="why-txt">The heartbeats in this recording could not be separated cleanly, so only the signal split is shown.</div>' +
        igChannelsOnly(d, tech);
    }
    var N = b.points, W = 300, H = 132, top = 16, shapeH = 58, attTop = 84, attH = 40;
    var bnd = b.phaseBounds, keys = ['upstroke', 'notch', 'decay'];
    var X = function (i) { return (i / (N - 1)) * W; };

    // Phase bands and labels
    var bands = '', labels = '';
    keys.forEach(function (k, j) {
      var x0 = X(bnd[j]), x1 = X(Math.min(bnd[j + 1], N - 1));
      bands += '<rect x="' + x0.toFixed(1) + '" y="0" width="' + (x1 - x0).toFixed(1) + '" height="' + H +
        '" fill="' + (j % 2 ? 'var(--wash-2)' : 'var(--wash-1)') + '"/>';
      labels += '<text x="' + ((x0 + x1) / 2).toFixed(1) + '" y="10" text-anchor="middle" font-size="7.5" font-weight="700" ' +
        'fill="var(--ink-400)" font-family="Inter,system-ui,sans-serif">' + BEAT_LAB[k][tech ? 1 : 0] + '</text>';
    });

    // Individual beats (faint) and their average (bold)
    var path = function (y) {
      return y.map(function (v, i) { return (i ? 'L' : 'M') + X(i).toFixed(1) + ',' + (top + (1 - v) * shapeH).toFixed(1); }).join('');
    };
    var faint = (b.beats || []).map(function (y) {
      return '<path d="' + path(y) + '" fill="none" stroke="var(--spo2)" stroke-opacity="0.16" stroke-width="1" vector-effect="non-scaling-stroke"/>';
    }).join('');
    var avg = '<path d="' + path(b.shape) + '" fill="none" stroke="var(--spo2)" stroke-width="2.2" vector-effect="non-scaling-stroke"/>';

    // Attention density: 1 = the attention an even spread would give.
    var dens = b.attention.map(function (a) { return a * N; });
    var sm = dens.map(function (_, i) {
      var t = 0, c = 0;
      for (var k = Math.max(0, i - 2); k <= Math.min(N - 1, i + 2); k++) { t += dens[k]; c++; }
      return t / c;
    });
    var mx = Math.max(2, Math.max.apply(null, sm) * 1.05);
    var Y = function (v) { return attTop + attH - (v / mx) * attH; };
    var area = 'M0,' + (attTop + attH) + sm.map(function (v, i) { return 'L' + X(i).toFixed(1) + ',' + Y(v).toFixed(1); }).join('') +
      'L' + W + ',' + (attTop + attH) + 'Z';
    var even = Y(1).toFixed(1);
    var att = '<path d="' + area + '" fill="var(--cyan-bright)" fill-opacity="0.35" stroke="var(--cyan-bright)" stroke-width="1.2" vector-effect="non-scaling-stroke"/>' +
      '<line x1="0" x2="' + W + '" y1="' + even + '" y2="' + even + '" stroke="var(--ink-400)" stroke-dasharray="3 3" stroke-width="1" vector-effect="non-scaling-stroke"/>' +
      '<text x="' + (W - 2) + '" y="' + (+even - 2) + '" text-anchor="end" font-size="6.5" fill="var(--ink-500)" font-family="Inter,system-ui,sans-serif">even spread</text>' +
      '<text x="2" y="' + (attTop - 3) + '" font-size="6.5" fill="var(--ink-500)" font-family="Inter,system-ui,sans-serif">' +
      (tech ? 'Attribution density (IG, per-beat normalised)' : 'Where the AI looked') + '</text>';

    var svg = '<svg class="why-beat" viewBox="0 0 ' + W + ' ' + H + '" role="img" ' +
      'aria-label="Average heartbeat with the AI\'s attention shown underneath">' + bands + labels + faint + avg + att + '</svg>';

    // Phase summary
    var ph = b.phases, best = null;
    keys.forEach(function (k) { if (!best || ph[k].density > ph[best].density) best = k; });
    var standOut = ph[best].density >= 1.15;
    var chips = keys.map(function (k) {
      var p = ph[k], hi = standOut && k === best;
      return '<div class="why-phase' + (hi ? ' hi' : '') + '"><div class="n">' + BEAT_LAB[k][tech ? 1 : 0] + '</div>' +
        '<div class="v">' + Math.round(p.attentionShare * 100) + '%</div>' +
        '<div class="s">of attention · ' + Math.round(p.timeShare * 100) + '% of the beat</div></div>';
    }).join('');
    var sentence = standOut
      ? (tech
          ? 'Attribution was densest on the <b>' + PHASE_TECH[best] + '</b>: ' + Math.round(ph[best].attentionShare * 100) +
            '% of it in ' + Math.round(ph[best].timeShare * 100) + '% of the beat (' + ph[best].density.toFixed(2) + '× an even spread).'
          : 'The AI paid extra attention to the <b>' + PHASE_NAME[best] + '</b>.')
      : (tech
          ? 'Attribution was spread close to evenly across the beat (no phase above 1.15× an even spread).'
          : 'The AI spread its attention fairly evenly across each heartbeat. No single part stood out.');
    var foot = tech
      ? 'Average of ' + b.nBeats + ' beats (~' + b.beatSeconds + ' s each). Each beat is weighted equally, which removes the model\'s ' +
        'tendency to weight the end of the recording more heavily.'
      : 'Based on ' + b.nBeats + ' heartbeats from this recording, layered on top of each other.';

    var cs = b.channelShare;
    var chBar = CH.map(function (c) {
      return '<span style="width:' + (100 * cs[c.k]).toFixed(1) + '%;background:' + c.color + ';"></span>';
    }).join('');
    var leg = CH.map(function (c) {
      return '<span><i style="background:' + c.color + '"></i>' + c.name + (tech ? ' (' + c.tech + ')' : '') +
        ' <b style="color:var(--ink-0)">' + Math.round(100 * cs[c.k]) + '%</b></span>';
    }).join('');

    return svg + '<div class="why-phases">' + chips + '</div>' +
      '<div class="why-txt">' + sentence + '</div>' +
      '<div class="why-muted" style="margin:2px 0 10px;">' + foot + '</div>' +
      '<div class="why-k2">Which signal it used</div>' +
      '<div class="why-ch">' + chBar + '</div><div class="why-leg">' + leg + '</div>';
  }

  function igChannelsOnly(d, tech) {
    var A = d.attribution, tot = { ppg: 0, vpg: 0, apg: 0 }, all = 0;
    CH.forEach(function (c) { (A[c.k] || []).forEach(function (v) { tot[c.k] += Math.abs(v); }); all += tot[c.k]; });
    all = all || 1;
    var chBar = CH.map(function (c) { return '<span style="width:' + (100 * tot[c.k] / all).toFixed(1) + '%;background:' + c.color + ';"></span>'; }).join('');
    var leg = CH.map(function (c) {
      return '<span><i style="background:' + c.color + '"></i>' + c.name + (tech ? ' (' + c.tech + ')' : '') +
        ' <b style="color:var(--ink-0)">' + Math.round(100 * tot[c.k] / all) + '%</b></span>';
    }).join('');
    return '<div class="why-k2">Which signal it used</div><div class="why-ch">' + chBar + '</div><div class="why-leg">' + leg + '</div>';
  }

  // Original 12-second view (no longer used on the page: its end-of-window bias misleads).
  function igHtmlWindow(d, tech) {
    var A = d.attribution, ppg = d.channels.ppg, n = ppg.length;
    var fs = (d.samplingRateHz || 125) / 5;             // backend downsamples by 5
    var tot = { ppg: 0, vpg: 0, apg: 0 }, imp = new Array(n).fill(0), all = 0;
    CH.forEach(function (c) {
      for (var i = 0; i < n; i++) { var a = Math.abs(A[c.k][i] || 0); tot[c.k] += a; imp[i] += a; }
      all += tot[c.k];
    });
    all = all || 1;

    // Which part of the beat got the most attention per unit time.
    var lab = beatPhases(ppg, fs), ph = {}, labelled = 0, labImp = 0;
    lab.forEach(function (l, i) {
      if (!l) return;
      ph[l] = ph[l] || { t: 0, a: 0 };
      ph[l].t++; ph[l].a += imp[i]; labelled++; labImp += imp[i];
    });
    var best = null;
    Object.keys(ph).forEach(function (k) {
      var dens = (ph[k].a / (labImp || 1)) / (ph[k].t / (labelled || 1));
      if (!best || dens > best.dens) best = { k: k, dens: dens, aShare: ph[k].a / (labImp || 1), tShare: ph[k].t / (labelled || 1) };
    });

    // Sparkline: PPG trace with attention bars underneath.
    var W = 300, H = 64, lo = Math.min.apply(null, ppg), hi = Math.max.apply(null, ppg), span = (hi - lo) || 1;
    var mx = Math.max.apply(null, imp) || 1, bars = '', line = '';
    for (var i = 0; i < n; i++) {
      var x = (i / (n - 1)) * W;
      var h = (imp[i] / mx) * (H - 6);
      if (h > 0.6) bars += '<rect x="' + x.toFixed(1) + '" y="' + (H - h).toFixed(1) + '" width="' + (W / n + 0.3).toFixed(2) +
        '" height="' + h.toFixed(1) + '" fill="var(--cyan-bright)" opacity="' + (0.25 + 0.75 * imp[i] / mx).toFixed(2) + '"/>';
      line += (i ? 'L' : 'M') + x.toFixed(1) + ',' + (4 + (1 - (ppg[i] - lo) / span) * (H - 26)).toFixed(1);
    }
    var svg = '<svg class="why-spark" viewBox="0 0 ' + W + ' ' + H + '" preserveAspectRatio="none" role="img" ' +
      'aria-label="Pulse waveform with the AI\'s attention shaded underneath">' + bars +
      '<path d="' + line + '" fill="none" stroke="var(--spo2)" stroke-width="1.6" vector-effect="non-scaling-stroke"/></svg>';

    var chBar = CH.map(function (c) {
      return '<span style="width:' + (100 * tot[c.k] / all).toFixed(1) + '%;background:' + c.color + ';"></span>';
    }).join('');
    var leg = CH.map(function (c) {
      return '<span><i style="background:' + c.color + '"></i>' + c.name + (tech ? ' (' + c.tech + ')' : '') + ' <b style="color:var(--ink-0)">' +
        Math.round(100 * tot[c.k] / all) + '%</b></span>';
    }).join('');

    var sentence = !best
      ? 'The heartbeats in this recording could not be separated cleanly, so only the overall split is shown.'
      : tech
        ? 'Attention was densest on the <b>' + PHASE_TECH[best.k] + '</b>: ' + Math.round(best.aShare * 100) +
          '% of attribution in ' + Math.round(best.tShare * 100) + '% of the time.'
        : 'The shaded bars show where the AI looked. It paid most attention to the <b>' + PHASE_NAME[best.k] + '</b>.';

    return svg + '<div class="why-txt">' + sentence + '</div>' +
      '<div class="why-ch">' + chBar + '</div><div class="why-leg">' + leg + '</div>';
  }

  // ---- SHAP factor summary ---------------------------------------------
  function shapHtml(s, tech) {
    var g = (s.magnitude && s.magnitude.groups) || [];
    if (!g.length) return '<div class="why-txt">No factor breakdown available for this window.</div>';
    var top = g.slice(0, 3), mx = Math.max.apply(null, top.map(function (x) { return Math.abs(x.mmHg); })) || 1;
    var rows = top.map(function (x) {
      var up = x.mmHg > 0, col = up ? 'var(--red)' : 'var(--green)';
      return '<div class="why-f" title="' + esc(x.what) + '"><div class="n">' + esc(x.label) + '</div>' +
        '<div class="v ' + (up ? 'up' : 'down') + '">' + (up ? '▲ +' : '▼ ') + x.mmHg.toFixed(1) + '</div>' +
        '<div class="bar"><span style="width:' + (100 * Math.abs(x.mmHg) / mx).toFixed(0) + '%;background:' + col + '"></span></div></div>';
    }).join('');
    var m = s.magnitude, ok = !m.agreement || m.agreement.close;
    var warn = ok ? '' : '<div class="why-muted" style="color:var(--amber)">Rough guide only: for this reading the factors don\'t fully account for the number.</div>';
    if (!tech) return rows + warn + '<div class="why-muted">How many mmHg each part of your pulse added to (▲) or took away from (▼) ' +
      'the top number, compared with the model\'s average reading.</div>';
    return rows + warn + '<div class="why-muted">SHAP on a surrogate trained to copy the deep model: mmHg each theme moved the systolic ' +
      'estimate from the average of ' + Math.round(m.averageEstimate) + ' to ' + Math.round(m.copyEstimate) +
      ' (deep model reading ' + Math.round(m.reading) + ').</div>';
  }

  // ---- Mount -----------------------------------------------------------
  var seq = 0;

  function mount(el, o) {
    if (!el) return;
    injectCss();
    var my = ++seq;
    el.__whySeq = my;
    var sbp = Math.round(o.sbp), dbp = Math.round(o.dbp), b = band(sbp, dbp);
    var q = '?target=SBP' + (o.window != null ? '&window=' + encodeURIComponent(o.window) : '');
    var fullHref = 'xai-results.html?patient=' + encodeURIComponent(o.key);
    // Doctors see method names and clinical terms; patients get plain language.
    var tech = o.technical != null ? !!o.technical : (function () {
      try { return localStorage.getItem('pulseiq_role') === 'doctor'; } catch (e) { return false; }
    })();

    el.innerHTML =
      '<div class="why ' + (o.layout === 'row' ? 'row' : 'stack') + (o.hideReading ? ' noread' : '') + '">' +
        '<div class="why-head"><h3>Why this reading</h3><span class="why-tag">Explainable AI</span>' +
          '<div class="grow"></div><a href="' + fullHref + '">Full explanation →</a></div>' +
        '<div class="why-grid">' +
          (o.hideReading ? '' : '<div class="why-sec"><div class="k">Reading</div>' +
            '<div class="why-band ' + b.cls + '">' + sbp + '/' + dbp + ' · ' + b.label + '</div>' +
            '<div class="why-txt">' + b.label + ' because ' + (tech ? b.why : b.why
              .replace('systolic <b>', 'the top number (<b>').replace('diastolic <b>', 'the bottom number (<b>')
              .replace(/<\/b>/g, '</b>)')) + '.</div>' +
            '<div class="why-muted">Uses the standard adult blood pressure ranges. This is an estimate, not a diagnosis.</div></div>') +
          '<div class="why-sec"><div class="k">What the AI looked at' + (tech ? ' <small>· deep model, Integrated Gradients</small>' : '') + '</div>' +
            '<div data-ig><div class="why-txt"><span class="why-spin"></span>Looking at your pulse…</div></div></div>' +
          '<div class="why-sec"><div class="k">What pushed it up or down' + (tech ? ' <small>· deep model, SHAP via surrogate</small>' : '') + '</div>' +
            '<div data-shap></div></div>' +
        '</div>' +
      '</div>';

    var igBox = el.querySelector('[data-ig]'), shapBox = el.querySelector('[data-shap]');

    fetch(API() + '/api/explain/saliency/' + encodeURIComponent(o.key) + q)
      .then(function (r) { if (!r.ok) throw new Error('bad'); return r.json(); })
      .then(function (d) { if (el.__whySeq === my) igBox.innerHTML = igHtml(d, tech); })
      .catch(function () { if (el.__whySeq === my) igBox.innerHTML = '<div class="why-txt">Not available right now: the PulseIQ server is offline.</div>'; });

    function loadShap() {
      shapBox.innerHTML = '<div class="why-txt"><span class="why-spin"></span>Working out the reasons. The first time takes about 15 seconds.</div>';
      fetch(API() + '/api/explain/summary/' + encodeURIComponent(o.key) + q)
        .then(function (r) { if (!r.ok) throw new Error('bad'); return r.json(); })
        .then(function (s) { if (el.__whySeq === my) shapBox.innerHTML = shapHtml(s, tech); })
        .catch(function () { if (el.__whySeq === my) shapBox.innerHTML = '<div class="why-txt">Not available right now: the PulseIQ server is offline.</div>'; });
    }

    if (o.shap === 'auto') loadShap();
    else {
      shapBox.innerHTML = '<div class="why-txt" style="margin-bottom:10px;">Which parts of the pulse pushed this reading up or down.</div>' +
        '<button class="why-btn" type="button">Show reasons</button>';
      shapBox.querySelector('button').onclick = loadShap;
    }
  }

  // v3: attention-only card (used once, on the Explainability page).
  function mountIG(el, o) {
    if (!el) return;
    injectCss();
    var tech = o.technical != null ? !!o.technical : (function () {
      try { return localStorage.getItem('pulseiq_role') === 'doctor'; } catch (e) { return false; }
    })();
    var q = '?target=' + encodeURIComponent(o.target || 'SBP') + (o.window != null ? '&window=' + encodeURIComponent(o.window) : '');
    var my = (el.__igSeq = (el.__igSeq || 0) + 1);
    el.innerHTML = '<div class="why-txt"><span class="why-spin"></span>Looking at the pulse…</div>';
    fetch(API() + '/api/explain/saliency/' + encodeURIComponent(o.key) + q)
      .then(function (r) { if (!r.ok) throw new Error('bad'); return r.json(); })
      .then(function (d) { if (el.__igSeq !== my) return; injectCss(); el.innerHTML = igHtml(d, tech); })
      .catch(function () { el.innerHTML = '<div class="why-txt">Not available right now: the PulseIQ server is offline.</div>'; });
  }

  // v3: one-line reason shown next to a reading, linking to the full page.
  function summaryLine(el, o) {
    if (!el) return;
    var href = 'xai-results.html?patient=' + encodeURIComponent(o.key) + (o.window != null ? '&window=' + o.window : '');
    var link = ' <a href="' + href + '" style="color:var(--cyan-bright);font-weight:700;white-space:nowrap;">See full explanation →</a>';
    el.innerHTML = '<span class="why-spin"></span>Working out the main reason…' + link;
    injectCss();
    var q = '?target=SBP' + (o.window != null ? '&window=' + encodeURIComponent(o.window) : '');
    fetch(API() + '/api/explain/summary/' + encodeURIComponent(o.key) + q)
      .then(function (r) { if (!r.ok) throw new Error('bad'); return r.json(); })
      .then(function (s) {
        var m = s.magnitude || {}, g = (m.groups || [])[0];
        if (m.agreement && !m.agreement.close) {
          el.innerHTML = 'No single clear reason stands out for this reading.' + link; return;
        }
        if (!g) { el.innerHTML = 'Reason not available for this reading.' + link; return; }
        var up = g.mmHg > 0;
        el.innerHTML = 'Main reason: <b>' + esc(g.label.toLowerCase()) + '</b> ' + (up ? 'pushed the top number up' : 'pulled the top number down') +
          ' by <b style="color:' + (up ? 'var(--red)' : 'var(--green)') + '">' + Math.abs(g.mmHg).toFixed(1) + ' mmHg</b>.' + link;
      })
      .catch(function () { el.innerHTML = 'Reason not available right now.' + link; });
  }

  window.PulseWhy = { mount: mount, mountIG: mountIG, summaryLine: summaryLine, band: band, igHtml: igHtml, injectCss: injectCss };
})();
