/* gqteashmd site: latest-release button, tabs, and the hero trajectory player. */

/* ---------- Latest release: point the button at the wheel ---------- */
(function () {
  var REPO = "AdemirCamargo57/gqteaSHMD";
  var btn = document.getElementById("download-btn");
  var label = document.getElementById("download-label");
  var meta = document.getElementById("download-meta");
  if (!btn || !window.fetch) return;

  function sizeKB(bytes) { return (bytes / 1024).toFixed(0) + " kB"; }
  function niceDate(iso) {
    try {
      return new Date(iso).toLocaleDateString("en-GB", { day: "numeric", month: "long", year: "numeric" });
    } catch (e) { return ""; }
  }

  fetch("https://api.github.com/repos/" + REPO + "/releases/latest", {
    headers: { Accept: "application/vnd.github+json" }
  })
    .then(function (r) { if (!r.ok) throw new Error(r.status); return r.json(); })
    .then(function (rel) {
      var assets = rel.assets || [];
      var wheel = assets.find(function (a) { return /\.whl$/i.test(a.name); });
      if (!wheel) return;
      var version = (rel.tag_name || "").replace(/^v/i, "");
      btn.href = rel.html_url || btn.href;
      label.textContent = "Get gqteashmd " + version;
      meta.textContent = "Linux only. Wheel " + sizeKB(wheel.size) + " and the lock file " +
        "environment.lock.txt, released " + niceDate(rel.published_at) +
        ". Free and open source, MIT license.";
    })
    .catch(function () { /* keep the fallback link to the latest release */ });
})();

/* ---------- Tabs (WAI-ARIA tabs pattern) ---------- */
(function () {
  var root = document.querySelector("[data-tabs]");
  if (!root) return;
  var tabs = Array.prototype.slice.call(root.querySelectorAll('[role="tab"]'));

  function select(tab, focus) {
    tabs.forEach(function (t) {
      var on = t === tab;
      t.setAttribute("aria-selected", on ? "true" : "false");
      t.tabIndex = on ? 0 : -1;
      document.getElementById(t.getAttribute("aria-controls")).hidden = !on;
    });
    if (focus) tab.focus();
  }

  tabs.forEach(function (tab, i) {
    tab.addEventListener("click", function () { select(tab, false); });
    tab.addEventListener("keydown", function (e) {
      var j = null;
      if (e.key === "ArrowRight") j = (i + 1) % tabs.length;
      else if (e.key === "ArrowLeft") j = (i - 1 + tabs.length) % tabs.length;
      else if (e.key === "Home") j = 0;
      else if (e.key === "End") j = tabs.length - 1;
      if (j !== null) { e.preventDefault(); select(tabs[j], true); }
    });
  });
})();

/* ---------- Hero trajectory: ethylene along one gqteashmd trajectory ---------- */
(function () {
  var canvas = document.getElementById("trajectory");
  var panel = document.getElementById("player");
  var fallback = document.getElementById("trajectory-fallback");
  var slider = document.getElementById("player-time");
  var button = document.getElementById("player-toggle");
  var readout = document.getElementById("player-readout");
  var marker = document.getElementById("player-hop");
  if (!canvas || !canvas.getContext || !window.fetch || !panel) return;
  var ctx = canvas.getContext("2d");

  var style = {
    C: { r: 0.42, c: "#5d6873" },
    H: { r: 0.26, c: "#f4f6f7" }
  };
  var cov = { C: 0.76, H: 0.31 };
  var FS_PER_SECOND = 10;  // playback speed: about 10 s per 100 fs
  var CAM = 30;            // camera distance for a gentle perspective
  var W = 560, H = 520;
  var dpr = Math.min(window.devicePixelRatio || 1, 2);
  canvas.width = W * dpr; canvas.height = H * dpr;

  var reduce = window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  var data = null, frame = 0, clock = 0, tEnd = 1, maxR = 1;
  var playing = !reduce, visible = true, last = 0;
  var rotY = -0.5, rotX = 0.35, dragging = false, lastX = 0, lastY = 0;

  function isDark() {
    var t = document.documentElement.getAttribute("data-theme");
    if (t) return t === "dark";
    return window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches;
  }

  function centred(k) {
    var f = data.positions_angstrom[k], n = f.length, cx = 0, cy = 0, cz = 0;
    f.forEach(function (a) { cx += a[0]; cy += a[1]; cz += a[2]; });
    cx /= n; cy /= n; cz /= n;
    return f.map(function (a, i) {
      return { el: data.symbols[i], x: a[0] - cx, y: a[1] - cy, z: a[2] - cz };
    });
  }

  function project(P) {
    var cy_ = Math.cos(rotY), sy = Math.sin(rotY), cx_ = Math.cos(rotX), sx = Math.sin(rotX);
    return P.map(function (p) {
      var x = p.x * cy_ + p.z * sy;
      var z = -p.x * sy + p.z * cy_;
      var y = p.y * cx_ - z * sx;
      z = p.y * sx + z * cx_;
      var persp = CAM / (CAM - z);
      return { x: x * persp, y: y * persp, z: z, s: persp, el: p.el };
    });
  }

  function shade(hex, f) {
    var n = parseInt(hex.slice(1), 16);
    var r = Math.round(((n >> 16) & 255) * f), g = Math.round(((n >> 8) & 255) * f), b = Math.round((n & 255) * f);
    return "rgb(" + r + "," + g + "," + b + ")";
  }

  function draw() {
    if (!data) return;
    var dark = isDark();
    var P = centred(frame);
    var Q = project(P);
    var scale = (Math.min(W, H) / 2) / (maxR * CAM / (CAM - maxR));
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, W, H);
    ctx.save();
    ctx.translate(W / 2, H / 2);

    // covalent bonds by distance, recomputed every frame (bonds may stretch)
    var items = [];
    for (var i = 0; i < P.length; i++) for (var j = i + 1; j < P.length; j++) {
      var dx = P[i].x - P[j].x, dy = P[i].y - P[j].y, dz = P[i].z - P[j].z;
      if (Math.sqrt(dx * dx + dy * dy + dz * dz) < 1.2 * (cov[P[i].el] + cov[P[j].el])) {
        items.push({ t: "b", z: (Q[i].z + Q[j].z) / 2 - 0.3, a: Q[i], c: Q[j] });
      }
    }
    Q.forEach(function (q) { items.push({ t: "a", z: q.z, q: q }); });
    items.sort(function (m, n) { return m.z - n.z; });

    items.forEach(function (it) {
      if (it.t === "b") {
        var w = 0.17 * scale * (it.a.s + it.c.s) / 2;
        ctx.lineCap = "round";
        ctx.lineWidth = w + 2.5;
        ctx.strokeStyle = dark ? "#0b1520" : "#3a4652";
        ctx.beginPath();
        ctx.moveTo(it.a.x * scale, -it.a.y * scale);
        ctx.lineTo(it.c.x * scale, -it.c.y * scale);
        ctx.stroke();
        ctx.lineWidth = w;
        ctx.strokeStyle = dark ? "#8796a6" : "#aab5bf";
        ctx.stroke();
      } else {
        var q = it.q, st = style[q.el] || style.C;
        var r = st.r * scale * q.s;
        var x = q.x * scale, y = -q.y * scale;
        var g = ctx.createRadialGradient(x - r * 0.35, y - r * 0.4, r * 0.1, x, y, r);
        g.addColorStop(0, "#ffffff");
        g.addColorStop(0.28, st.c);
        g.addColorStop(1, shade(st.c, 0.55));
        ctx.beginPath();
        ctx.arc(x, y, r, 0, Math.PI * 2);
        ctx.fillStyle = g;
        ctx.fill();
        ctx.lineWidth = 1.5;
        ctx.strokeStyle = dark ? "#0b1520" : "#33404d";
        ctx.stroke();
      }
    });
    ctx.restore();
  }

  function frameAt(t) {
    var k = 0;
    while (k + 1 < data.time_fs.length && data.time_fs[k + 1] <= t) k++;
    return k;
  }

  function show(k) {
    frame = k;
    slider.value = String(k);
    var labels = data.state_labels, h = data.hop_index;
    var state = labels[data.active[k]];
    var text = state;
    if (k === h) text = labels[data.active[h - 1]] + " → " + state;
    readout.textContent = "t = " + data.time_fs[k].toFixed(1) + " fs · " + text;
    readout.setAttribute("data-state", state);
    draw();
  }

  function setPlaying(on) {
    playing = on;
    button.textContent = on ? "Pause" : "Play";
    button.setAttribute("aria-label", on ? "Pause the trajectory" : "Play the trajectory");
    if (on) { last = 0; requestAnimationFrame(tick); }
  }

  function tick(t) {
    if (!playing || !visible) { last = 0; return; }
    var dt = last ? Math.min(t - last, 50) : 16;
    last = t;
    clock += dt * FS_PER_SECOND / 1000;
    if (clock > tEnd) clock = 0;
    var k = frameAt(clock);
    if (k !== frame) show(k); else if (dragging) draw();
    requestAnimationFrame(tick);
  }

  function setup(d) {
    data = d;
    tEnd = data.time_fs[data.time_fs.length - 1];
    for (var k = 0; k < data.positions_angstrom.length; k++) {
      centred(k).forEach(function (p) {
        maxR = Math.max(maxR, Math.sqrt(p.x * p.x + p.y * p.y + p.z * p.z) + 0.45);
      });
    }
    slider.max = String(data.time_fs.length - 1);
    marker.style.left = (100 * data.time_fs[data.hop_index] / tEnd) + "%";
    if (fallback) fallback.hidden = true;
    panel.hidden = false;
    show(0);
    setPlaying(playing);
  }

  fetch("trajectory.json")
    .then(function (r) { if (!r.ok) throw new Error(r.status); return r.json(); })
    .then(setup)
    .catch(function () { /* the fallback text stays */ });

  slider.addEventListener("input", function () {
    if (playing) setPlaying(false);
    var k = parseInt(slider.value, 10);
    clock = data.time_fs[k];
    show(k);
  });
  button.addEventListener("click", function () { setPlaying(!playing); });

  if ("IntersectionObserver" in window) {
    new IntersectionObserver(function (entries) {
      var was = visible;
      visible = entries[0].isIntersecting;
      if (visible && !was && playing) { last = 0; requestAnimationFrame(tick); }
    }).observe(canvas);
  }

  canvas.addEventListener("pointerdown", function (e) {
    dragging = true; lastX = e.clientX; lastY = e.clientY;
    canvas.setPointerCapture(e.pointerId);
  });
  canvas.addEventListener("pointermove", function (e) {
    if (!dragging) return;
    rotY += (e.clientX - lastX) * 0.01;
    rotX += (e.clientY - lastY) * 0.01;
    rotX = Math.max(-1.4, Math.min(1.4, rotX));
    lastX = e.clientX; lastY = e.clientY;
    draw();
  });
  function end() { dragging = false; }
  canvas.addEventListener("pointerup", end);
  canvas.addEventListener("pointercancel", end);

  if (window.matchMedia) {
    var mq = window.matchMedia("(prefers-color-scheme: dark)");
    if (mq.addEventListener) mq.addEventListener("change", draw);
  }
})();
