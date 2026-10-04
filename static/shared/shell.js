/* ============================================================
   PulseIQ — shared app shell (labeled sidebar)
   Renders the same left sidebar on every dashboard-style page so
   navigation and iconography can never drift between pages.
   Each page just needs: <div id="rail" data-active="home"></div>
   ============================================================ */
(function () {
  // v3: every call to the PulseIQ backend carries the session token, and an
  // expired or missing session sends the user back to the login page instead
  // of silently showing someone else's data.
  (function () {
    var api = window.PULSEIQ_API_BASE || (location.port === '8090' ? 'http://localhost:8001' : location.origin);
    var page = (location.pathname.split('/').pop() || 'index.html').toLowerCase();
    var PUBLIC = ['index.html', 'login.html', ''];
    var isPublic = PUBLIC.indexOf(page) > -1;
    if (!isPublic && !localStorage.getItem('pulseiq_token')) { location.replace('login.html'); return; }
    var nativeFetch = window.fetch.bind(window);
    window.fetch = function (input, init) {
      var url = typeof input === 'string' ? input : (input && input.url) || '';
      if (url.indexOf(api) === 0) {
        init = init || {};
        var h = new Headers(init.headers || {});
        var tok = localStorage.getItem('pulseiq_token');
        if (tok && !h.has('Authorization')) h.set('Authorization', 'Bearer ' + tok);
        init.headers = h;
        return nativeFetch(input, init).then(function (r) {
          if (r.status === 401 && !isPublic && url.indexOf('/api/auth/') === -1) {
            localStorage.removeItem('pulseiq_token');
            location.replace('login.html');
          }
          return r;
        });
      }
      return nativeFetch(input, init);
    };
  })();

  var role = localStorage.getItem('pulseiq_role') || 'patient';
  var name = localStorage.getItem('pulseiq_name') || (role === 'doctor' ? 'Dr. Demo' : 'Demo Patient');
  var initials = name.split(' ').map(function (w) { return w[0]; }).join('').slice(0, 2).toUpperCase();
  var isRealAccount = !!localStorage.getItem('pulseiq_token');
  var roleLabel = role + (isRealAccount ? '' : ' · guest');

  // Destinations used to live in the sidebar. They now render as tiles on the
  // dashboard instead — the sidebar is reserved for account controls only.
  var HOME = { patient: 'patient-dashboard.html', doctor: 'doctor-dashboard.html' };

  var HUB = {
    patient: [
      { key: 'monitor',   href: 'icu-monitor.html',        icon: '\ud83e\udec0', label: 'Live Monitor',   sub: 'Real-time waveform' },
      { key: 'history',   href: 'history.html',            icon: '\ud83d\udcdc', label: 'History',        sub: 'Trends & stats' },
      { key: 'insights',  href: 'xai-results.html',        icon: '\ud83e\udde0', label: 'Explainability', sub: 'Why this reading' },
      { key: 'medicines', href: 'medicine-reminders.html', icon: '\ud83d\udc8a', label: 'Medicines',      sub: 'Doses & adherence' },
      { key: 'finder',    href: 'hospital-finder.html',    icon: '\ud83d\uddfa\ufe0f', label: 'Find Care', sub: 'Doctors & hospitals' },
      { key: 'vault',     href: 'medical-vault.html',      icon: '\ud83d\uddc2\ufe0f', label: 'My Vault',  sub: 'Reports & scans' },
      { key: 'chat',      href: 'chat.html',               icon: '\ud83d\udcac', label: 'Messages',       sub: 'Doctors & AI assistant' },
      { key: 'upload',    href: 'upload.html',             icon: '\ud83d\udce4', label: 'Upload PPG',     sub: 'Test your own recording' }
    ],
    doctor: [
      { key: 'monitor',  href: 'icu-monitor.html', icon: '\ud83e\udec0', label: 'ICU Monitor',      sub: 'Patient waveforms' },
      { key: 'insights', href: 'xai-results.html', icon: '\ud83e\udde0', label: 'Explainability',   sub: 'Why this reading' },
      { key: 'chat',     href: 'chat.html',        icon: '\ud83d\udcac', label: 'Patient Messages', sub: 'Your patients' },
      { key: 'upload',   href: 'upload.html',      icon: '\ud83d\udce4', label: 'Upload PPG',       sub: 'Analyse a recording' }
    ]
  };

  var FOOT = [
    { key: 'settings', href: 'settings.html', icon: '\u2699\ufe0f', label: 'Settings' }
  ];

  function homeHref() { return HOME[role] || HOME.patient; }

  function wireAccountMenu() {
    var btn = document.getElementById('pulseAcct');
    var pop = document.getElementById('pulseAcctPop');
    if (!btn || !pop) return;
    var open = false;
    btn.addEventListener('click', function (e) {
      e.stopPropagation();
      open = !open;
      pop.classList.toggle('on', open);
      btn.classList.toggle('on', open);
    });
    pop.addEventListener('click', function (e) { e.stopPropagation(); });
    document.addEventListener('click', function () {
      open = false; pop.classList.remove('on'); btn.classList.remove('on');
    });
    function row(k, v, cls) {
      return '<div class="ap-r"><span class="k">' + k + '</span>' +
             '<span class="v' + (cls ? ' ' + cls : '') + '">' + v + '</span></div>';
    }
    var grid = document.getElementById('pulseAcctGrid');
    // A linked recording is a patient-only concept -- a clinician's account is
    // never tied to one, so the row is omitted for doctors rather than shown
    // permanently reading "not linked".
    function recordingRow(subj) {
      if (role === 'doctor') return '';
      return row('Recording', subj ? subj : 'not linked', subj ? '' : 'none');
    }
    // Real account details only -- nothing here is filled in from a guess.
    var token = localStorage.getItem('pulseiq_token');
    if (!token) {
      grid.innerHTML = row('Role', role) +
        (role === 'doctor' ? '' : row('Recording', 'demo data', 'none'));
      return;
    }
    var api = window.PULSEIQ_API_BASE || (location.port === '8090' ? 'http://localhost:8001' : location.origin);
    fetch(api + '/api/auth/me', { headers: { Authorization: 'Bearer ' + token } })
      .then(function (r) { return r.ok ? r.json() : null; })
      .then(function (d) {
        if (!d) { grid.innerHTML = row('Account', 'session expired', 'none'); return; }
        document.getElementById('pulseAcctEmail').textContent = d.user.email || '';
        grid.innerHTML = row('Role', d.user.role) + recordingRow(d.user.subject_id);
      })
      .catch(function () {
        document.getElementById('pulseAcctEmail').textContent = 'backend offline';
        grid.innerHTML = row('Role', role) +
          (role === 'doctor' ? '' : row('Recording', 'unavailable', 'none'));
      });
  }

  function renderRail() {
    var el = document.getElementById('rail');
    if (!el) return;
    var active = el.getAttribute('data-active') || 'home';
    // The logo is the way home now that page links live on the dashboard.
    var html = '<a href="' + homeHref() + '" class="r-logo" style="text-decoration:none;" title="Back to dashboard">'
      + '<div class="mark"></div><div><h2>Pulse<span>IQ</span></h2>'
      + '<div class="r-home">AI Cuffless BP Monitoring</div></div></a>';
    // v3: wrist-sensor artwork fills the quiet middle of the sidebar
    html += '<div class="rail-art" aria-hidden="true"><img src="shared/img/hand-rail.webp" alt=""><i class="rail-pulse"></i>'
         +  '<div class="rail-cap"><b>Pulse in.</b> Pressure out.</div></div>';
    html += '<div class="spacer"></div>';
    FOOT.forEach(function (it) {
      html += '<a class="icon' + (it.key === active ? ' on' : '') + '" href="' + it.href + '"><span class="ic">' + it.icon + '</span>' + it.label + '</a>';
    });
    html += '<div class="r-foot">';
    html += '<a class="icon" href="login.html" id="pulseLogoutLink"><span class="ic">\u21aa</span>Logout</a>';
    html += '<div class="r-acct">';
    html += '<div class="r-user" id="pulseAcct" title="Account"><div class="avatar">' + initials + '</div>'
         +  '<div><div class="nm">' + name + '</div><div class="rl">' + roleLabel + '</div></div>'
         +  '<span class="chev">\u203a</span></div>';
    // The card itself is the profile summary -- no menu rows. Clicking it opens
    // the full profile page; Logout already has its own row above.
    html += '<a class="acct-pop" id="pulseAcctPop" href="' + (isRealAccount ? 'profile.html' : 'login.html') + '">'
         +  '<div class="ap-head"><div class="ap-av">' + initials + '</div>'
         +  '<div><div class="ap-n">' + name + '</div>'
         +  '<div class="ap-e" id="pulseAcctEmail">' + (isRealAccount ? 'Loading\u2026' : 'Not signed in') + '</div>'
         +  '</div></div>'
         +  '<div class="ap-grid" id="pulseAcctGrid"></div>'
         +  '<div class="ap-go">' + (isRealAccount ? 'Open full profile \u2192' : 'Sign in \u2192') + '</div>'
         +  '</a>';
    html += '</div>';
    html += '</div>';
    el.innerHTML = html;
    wireAccountMenu();
    wireMobileDrawer(el);
    var logoutLink = document.getElementById('pulseLogoutLink');
    if (logoutLink) {
      logoutLink.addEventListener('click', function () {
        localStorage.removeItem('pulseiq_role');
        localStorage.removeItem('pulseiq_name');
        localStorage.removeItem('pulseiq_token');
      });
    }
  }

  // ---- Theme -------------------------------------------------------------
  // The choice is applied by a tiny inline script in each page's <head> so the
  // first paint is already correct; this only handles switching afterwards.
  var THEME_KEY = 'pulseiq_theme';

  function currentTheme() {
    return document.documentElement.getAttribute('data-theme') === 'light' ? 'light' : 'dark';
  }

  function applyTheme(mode) {
    if (mode === 'light') document.documentElement.setAttribute('data-theme', 'light');
    else document.documentElement.removeAttribute('data-theme');
    try { localStorage.setItem(THEME_KEY, mode); } catch (e) {}
    // Tell the address bar / status bar to match, so the phone chrome doesn't
    // stay black around a white page.
    var meta = document.querySelector('meta[name="theme-color"]');
    if (!meta) {
      meta = document.createElement('meta');
      meta.setAttribute('name', 'theme-color');
      document.head.appendChild(meta);
    }
    meta.setAttribute('content', mode === 'light' ? '#eef2f7' : '#04060a');
    document.querySelectorAll('[data-theme-btn]').forEach(function (b) {
      b.innerHTML = mode === 'light' ? SUN : MOON;
      b.setAttribute('aria-label', mode === 'light' ? 'Switch to dark mode' : 'Switch to light mode');
      b.setAttribute('title', mode === 'light' ? 'Dark mode' : 'Light mode');
    });
  }

  // Inline SVG rather than an emoji: emoji render in colour and at wildly
  // different sizes per platform, which looked wrong next to the other icons.
  var MOON = '<svg viewBox="0 0 24 24" width="17" height="17" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round"><path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z"/></svg>';
  var SUN  = '<svg viewBox="0 0 24 24" width="17" height="17" fill="none" stroke="currentColor" stroke-width="1.9" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="4.2"/><path d="M12 2.6v2M12 19.4v2M4.2 4.2l1.4 1.4M18.4 18.4l1.4 1.4M2.6 12h2M19.4 12h2M4.2 19.8l1.4-1.4M18.4 5.6l1.4-1.4"/></svg>';

  // Every page has a different header element, so the button is placed in
  // whichever one exists rather than being duplicated into 13 files.
  function mountThemeToggle() {
    if (document.querySelector('[data-theme-btn]')) return;
    var host = document.querySelector('.topbar')      // app shell pages
            || document.querySelector('.finder-top')  // map
            || document.querySelector('.icu-top')     // monitor
            || document.querySelector('.nav')         // landing
            || document.querySelector('.stage');      // login
    if (!host) return;

    var btn = document.createElement('button');
    btn.className = 'theme-btn';
    btn.setAttribute('data-theme-btn', '1');
    btn.setAttribute('type', 'button');
    btn.addEventListener('click', function () {
      applyTheme(currentTheme() === 'light' ? 'dark' : 'light');
    });
    host.appendChild(btn);           // last child = top right of every header
    applyTheme(currentTheme());
  }

  // On the pages with no shell, renderRail() never runs, so mount directly.
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', mountThemeToggle);
  } else {
    mountThemeToggle();
  }

  // A "Back" button on every inner page (the dashboards, landing, login and the
  // live monitor, which has its own, are left alone). Goes to the previous page
  // if it was part of PulseIQ, otherwise to the user's dashboard.
  function mountBackButton() {
    var page = (location.pathname.split('/').pop() || 'index.html').toLowerCase();
    var skip = ['', 'index.html', 'login.html', 'patient-dashboard.html', 'doctor-dashboard.html', 'icu-monitor.html'];
    if (skip.indexOf(page) >= 0 || document.querySelector('[data-back-btn]')) return;
    var host = document.querySelector('.topbar') || document.querySelector('.finder-top');
    if (!host) return;
    var a = document.createElement('a');
    a.className = 'back-btn';
    a.setAttribute('data-back-btn', '1');
    a.href = homeHref();
    a.textContent = '\u2190 Back';
    a.addEventListener('click', function (e) {
      var ref = document.referrer || '';
      if (ref && ref.indexOf(location.origin) === 0 && history.length > 1 && ref !== location.href) {
        e.preventDefault(); history.back();
      }
    });
    host.insertBefore(a, host.firstChild);
  }
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', mountBackButton);
  } else {
    mountBackButton();
  }

  // On a phone the rail is an off-canvas drawer. The button lives in the page's
  // own topbar, so it is injected here rather than duplicated in 13 files.
  function wireMobileDrawer(rail) {
    var bar = document.querySelector('.topbar');
    if (!bar) return;

    var scrim = document.querySelector('.rail-scrim');
    if (!scrim) {
      scrim = document.createElement('div');
      scrim.className = 'rail-scrim';
      document.body.appendChild(scrim);
    }

    var burger = document.getElementById('railBurger');
    if (!burger) {
      burger = document.createElement('button');
      burger.id = 'railBurger';
      burger.className = 'rail-burger';
      burger.setAttribute('aria-label', 'Open menu');
      burger.setAttribute('aria-expanded', 'false');
      burger.innerHTML = '\u2630';
      bar.insertBefore(burger, bar.firstChild);
    }

    function setOpen(open) {
      rail.classList.toggle('open', open);
      scrim.classList.toggle('on', open);
      burger.setAttribute('aria-expanded', open ? 'true' : 'false');
      // Stop the page behind the drawer from scrolling under a thumb.
      document.body.style.overflow = open ? 'hidden' : '';
    }

    burger.addEventListener('click', function (e) {
      e.stopPropagation();
      setOpen(!rail.classList.contains('open'));
    });
    scrim.addEventListener('click', function () { setOpen(false); });
    // Following a link inside the drawer should not leave it open behind the
    // next page's paint.
    rail.addEventListener('click', function (e) {
      if (e.target.closest('a')) setOpen(false);
    });
    document.addEventListener('keydown', function (e) {
      if (e.key === 'Escape') setOpen(false);
    });
    // Rotating to landscape can cross the breakpoint; drop the drawer state so
    // the desktop layout never starts with body scroll locked.
    window.addEventListener('resize', function () {
      if (window.innerWidth > 900) setOpen(false);
    });
  }

  // Dashboard tile grid — the single source of truth for where a user can go,
  // so the patient and doctor dashboards can never drift out of sync.
  // v3: line icons from the BP visual kit when it is loaded; emoji otherwise.
  var HUB_ICON = { monitor: 'monitor', history: 'bp', insights: 'brain', medicines: 'pill',
                   finder: 'pin', vault: 'file', chat: 'chat' };
  function hubIcon(it) {
    return (window.PulseArt && HUB_ICON[it.key]) ? window.PulseArt.icon(HUB_ICON[it.key], 26) : it.icon;
  }
  function renderHub(elId) {
    var el = document.getElementById(elId || 'qaHub');
    if (!el) return;
    var items = HUB[role] || HUB.patient;
    // 7 tiles over a 4-wide grid leaves a hole in the last row; widening the
    // final tile to fill it keeps the block rectangular at any count.
    var cols = window.innerWidth <= 820 ? 2 : (window.innerWidth <= 1180 ? 3 : 4);
    var rem = items.length % cols;
    var lastSpan = rem === 0 ? 1 : (cols - rem + 1);
    el.innerHTML = items.map(function (it, i) {
      var span = (i === items.length - 1 && lastSpan > 1) ? 'grid-column:span ' + lastSpan + ';' : '';
      return '<a class="qa-tile" href="' + it.href + '" style="text-decoration:none;position:relative;' + span + '"' +
             (it.key === 'chat' ? ' data-unread-slot="1"' : '') + '>' +
             '<div class="ic ic-' + it.key + '">' + hubIcon(it) + '</div>' +
             '<div class="t">' + it.label + '</div>' +
             '<div class="d">' + it.sub + '</div></a>';
    }).join('');
  }

  function currentRole() { return role; }
  function currentName() { return name; }

  // ---- Unread-message badge on the Messages nav item -------------------
  // Rendered on every page so a new message is visible from anywhere, not
  // only when the chat page happens to be open.
  function setNavUnread(count) {
    var links = document.querySelectorAll('[data-unread-slot], #rail a[href="chat.html"]');
    Array.prototype.forEach.call(links, function (link) {
      var badge = link.querySelector('.nav-unread');
      if (!count || count < 1) { if (badge) badge.remove(); return; }
      if (!badge) {
        badge = document.createElement('span');
        badge.className = 'nav-unread';
        badge.style.cssText = 'position:absolute;top:8px;right:8px;min-width:18px;height:18px;padding:0 6px;' +
          'border-radius:99px;background:linear-gradient(90deg,#22e5f0,#7bf5fb);color:#04070b;font-size:10px;' +
          'font-weight:800;display:flex;align-items:center;justify-content:center;' +
          'box-shadow:0 0 12px -2px rgba(34,229,240,0.7);';
        link.appendChild(badge);
      }
      badge.textContent = count > 99 ? '99+' : count;
    });
  }

  function pollUnread() {
    var token = localStorage.getItem('pulseiq_token');
    if (!token) return;                       // guests have no real inbox
    var api = window.PULSEIQ_API_BASE || (location.port === '8090' ? 'http://localhost:8001' : location.origin);
    fetch(api + '/api/unread', { headers: { Authorization: 'Bearer ' + token } })
      .then(function (r) { return r.ok ? r.json() : null; })
      .then(function (d) { if (d) setNavUnread(d.totalUnread); })
      .catch(function () { /* backend down — just leave the badge as-is */ });
  }

  // ---- Which recording belongs to the signed-in patient ----------------
  // Guests fall back to the demo subject; a real account uses only the
  // recording it has been linked to, and gets null when it has none — so no
  // page ever shows someone else's data under the wrong name.
  var DEMO_SUBJECT = 'rohit-sharma';

  function resolvePatientKey() {
    var api = window.PULSEIQ_API_BASE || (location.port === '8090' ? 'http://localhost:8001' : location.origin);
    var token = localStorage.getItem('pulseiq_token');
    if (!token) {
      return Promise.resolve({ key: null, linked: false, guest: true, name: null });
    }
    return fetch(api + '/api/auth/me', { headers: { Authorization: 'Bearer ' + token } })
      .then(function (r) { return r.ok ? r.json() : null; })
      .then(function (d) {
        if (!d) return { key: null, linked: false, guest: true, name: null };
        var subj = d.user.subject_id;
        return {
          key: subj || null,
          linked: !!subj,
          guest: false,
          name: d.user.name,
          userId: d.user.id
        };
      })
      .catch(function () {
        return { key: null, linked: false, guest: true, name: null };
      });
  }

  // Standard "your account has no recording yet" panel.
  function noDataHtml(name) {
    return '<div class="card" style="text-align:center;padding:42px 28px;">' +
      '<div style="font-size:34px;margin-bottom:10px;">📉</div>' +
      '<h3 style="margin:0 0 8px;">No recording linked to ' + (name ? name : 'this account') + '</h3>' +
      '<div style="font-size:12.5px;color:var(--ink-400);line-height:1.7;max-width:460px;margin:0 auto 18px;">' +
      'This account has no PPG recording yet, so there are no readings to show. ' +
      'Enter the Patient ID of a recording to link it to your account.</div>' +
      '<a href="profile.html" style="display:inline-block;background:linear-gradient(90deg,#22e5f0,#7bf5fb);' +
      'color:#04070b;padding:11px 22px;border-radius:11px;font-weight:800;font-size:12.5px;text-decoration:none;">' +
      'Link a Patient ID →</a></div>';
  }

  window.PulseShell = {
    renderRail: renderRail, renderHub: renderHub, homeHref: homeHref,
    role: currentRole, name: currentName,
    setNavUnread: setNavUnread, pollUnread: pollUnread,
    resolvePatientKey: resolvePatientKey, noDataHtml: noDataHtml,
    DEMO_SUBJECT: DEMO_SUBJECT
  };
  document.addEventListener('DOMContentLoaded', function () {
    renderRail();
    renderHub('qaHub');
    var rT;
    window.addEventListener('resize', function () {
      clearTimeout(rT);
      rT = setTimeout(function () { renderHub('qaHub'); pollUnread(); }, 180);
    });
    pollUnread();
    setInterval(pollUnread, 10000);
  });
})();
