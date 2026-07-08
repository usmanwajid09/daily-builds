// Vanilla JS frontend for football-stats-site. No build step, no
// framework -- matches the backend's stdlib-only convention. Talks to
// the JSON API under /api/* (same origin, served by CombinedApp).
(function () {
  "use strict";

  const API = {
    async get(path) {
      const res = await fetch(path);
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body.error || `${res.status} ${res.statusText}`);
      }
      return res.json();
    },
    standings: () => API.get("/api/standings"),
    matches: () => API.get("/api/matches"),
    live: () => API.get("/api/live"),
    topScorers: (limit) => API.get(`/api/top-scorers${limit ? `?limit=${limit}` : ""}`),
    team: (name) => API.get(`/api/teams/${encodeURIComponent(name)}`),
    search: (q) => API.get(`/api/search?q=${encodeURIComponent(q)}`),
  };

  const $ = (sel) => document.querySelector(sel);
  const el = (tag, attrs = {}, children = []) => {
    const node = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
      if (k === "class") node.className = v;
      else if (k === "text") node.textContent = v;
      else if (k.startsWith("on") && typeof v === "function") node.addEventListener(k.slice(2), v);
      else node.setAttribute(k, v);
    }
    for (const child of [].concat(children)) {
      if (child) node.appendChild(typeof child === "string" ? document.createTextNode(child) : child);
    }
    return node;
  };

  // ---- Tab navigation -----------------------------------------------

  const views = {
    table: $("#view-table"),
    matches: $("#view-matches"),
    live: $("#view-live"),
    scorers: $("#view-scorers"),
    team: $("#view-team"),
  };
  let livePollTimer = null;

  function showTab(name) {
    for (const key of Object.keys(views)) {
      views[key].classList.toggle("active", key === name);
    }
    document.querySelectorAll(".tab-btn").forEach((btn) => {
      btn.classList.toggle("active", btn.dataset.tab === name);
    });
    stopLivePolling();
    if (name === "table") loadStandings();
    if (name === "matches") loadMatches();
    if (name === "live") startLivePolling();
    if (name === "scorers") loadScorers();
  }

  document.querySelectorAll(".tab-btn").forEach((btn) => {
    btn.addEventListener("click", () => showTab(btn.dataset.tab));
  });

  function showTeam(name) {
    for (const key of Object.keys(views)) views[key].classList.toggle("active", key === "team");
    document.querySelectorAll(".tab-btn").forEach((b) => b.classList.remove("active"));
    stopLivePolling();
    loadTeam(name);
  }
  $("#team-back").addEventListener("click", () => showTab("table"));

  // ---- League table ---------------------------------------------------

  async function loadStandings() {
    const wrap = $("#table-wrap");
    wrap.className = "loading";
    wrap.textContent = "Loading standings…";
    try {
      const data = await API.standings();
      wrap.className = "";
      wrap.innerHTML = "";
      wrap.appendChild(renderStandingsTable(data.standings));
    } catch (err) {
      wrap.className = "error";
      wrap.textContent = `Couldn't load standings: ${err.message}`;
    }
  }

  function renderStandingsTable(rows) {
    const table = el("table", { class: "standings" });
    table.appendChild(el("thead", {}, el("tr", {}, [
      el("th", { text: "#" }),
      el("th", { class: "team-col", text: "Team" }),
      el("th", { text: "P" }),
      el("th", { text: "W" }),
      el("th", { text: "D" }),
      el("th", { text: "L" }),
      el("th", { text: "GF" }),
      el("th", { text: "GA" }),
      el("th", { text: "GD" }),
      el("th", { text: "Pts" }),
    ])));
    const tbody = el("tbody");
    for (const row of rows) {
      const tr = el("tr", { onclick: () => showTeam(row.team) }, [
        el("td", { text: String(row.position) }),
        el("td", { class: "team-cell" }, el("a", { class: "team-link", href: "#", text: row.team, onclick: (e) => e.preventDefault() })),
        el("td", { text: String(row.played) }),
        el("td", { text: String(row.won) }),
        el("td", { text: String(row.drawn) }),
        el("td", { text: String(row.lost) }),
        el("td", { text: String(row.goals_for) }),
        el("td", { text: String(row.goals_against) }),
        el("td", { text: String(row.goal_difference) }),
        el("td", { class: "pts", text: String(row.points) }),
      ]);
      tbody.appendChild(tr);
    }
    table.appendChild(tbody);
    return table;
  }

  // ---- Fixtures & results ---------------------------------------------

  let allMatchesCache = null;

  async function loadMatches() {
    const wrap = $("#matches-wrap");
    wrap.className = "loading";
    wrap.textContent = "Loading matches…";
    try {
      if (!allMatchesCache) {
        const data = await API.matches();
        allMatchesCache = data.matches;
        populateMatchdayFilter(allMatchesCache);
      }
      renderMatches(currentMatchdayFilter());
    } catch (err) {
      wrap.className = "error";
      wrap.textContent = `Couldn't load matches: ${err.message}`;
    }
  }

  function currentMatchdayFilter() {
    const val = $("#matchday-filter").value;
    return val ? parseInt(val, 10) : null;
  }

  function populateMatchdayFilter(matches) {
    const select = $("#matchday-filter");
    if (select.dataset.populated) return;
    const matchdays = [...new Set(matches.map((m) => m.matchday))].sort((a, b) => a - b);
    for (const md of matchdays) {
      select.appendChild(el("option", { value: String(md), text: `Matchday ${md}` }));
    }
    select.dataset.populated = "1";
    select.addEventListener("change", () => renderMatches(currentMatchdayFilter()));
  }

  function renderMatches(matchdayFilter) {
    const wrap = $("#matches-wrap");
    wrap.className = "";
    wrap.innerHTML = "";
    let matches = allMatchesCache;
    if (matchdayFilter !== null) matches = matches.filter((m) => m.matchday === matchdayFilter);

    if (matches.length === 0) {
      wrap.appendChild(el("div", { class: "empty", text: "No matches for this filter." }));
      return;
    }

    const byMatchday = {};
    for (const m of matches) {
      (byMatchday[m.matchday] = byMatchday[m.matchday] || []).push(m);
    }
    const matchdays = Object.keys(byMatchday).map(Number).sort((a, b) => a - b);
    for (const md of matchdays) {
      wrap.appendChild(el("h3", { class: "matchday-heading", text: `Matchday ${md}` }));
      const list = el("div", { class: "match-list" });
      for (const m of byMatchday[md]) list.appendChild(renderMatchCard(m));
      wrap.appendChild(list);
    }
  }

  function renderMatchCard(m) {
    const played = m.status === "FT";
    const scoreText = played ? `${m.home_score} – ${m.away_score}` : "vs";
    const badgeClass = played ? "ft" : (m.status === "LIVE" ? "live" : "");
    const card = el("div", { class: "match-card" }, [
      el("div", { class: "team home" }, el("span", { class: "team-name", text: m.home_team })),
      el("div", { class: "score-block" }, [
        el("div", { class: "score", text: scoreText }),
        el("div", { class: "kickoff", text: played ? m.date : `${m.date} · ${m.kickoff}` }),
        el("div", { class: `status-badge ${badgeClass}`, text: m.status }),
      ]),
      el("div", { class: "team away" }, el("span", { class: "team-name", text: m.away_team })),
    ]);
    if (m.scorers && m.scorers.length) {
      const text = m.scorers
        .slice().sort((a, b) => a.minute - b.minute)
        .map((s) => `${s.player || "?"} ${s.minute}'`)
        .join(", ");
      card.appendChild(el("div", { class: "scorers", text }));
    }
    return card;
  }

  // ---- Live -------------------------------------------------------------

  function startLivePolling() {
    pollLiveOnce();
    livePollTimer = setInterval(pollLiveOnce, 3000);
  }
  function stopLivePolling() {
    if (livePollTimer) {
      clearInterval(livePollTimer);
      livePollTimer = null;
    }
  }

  async function pollLiveOnce() {
    const wrap = $("#live-wrap");
    try {
      const data = await API.live();
      $("#live-clock").textContent = data.matches.length
        ? `${Math.min(data.clock_minute, 90)}'`
        : "";
      $("#live-dot").classList.toggle("hidden", !(data.matches.length && !data.finished));

      wrap.className = "";
      if (!data.matches.length) {
        wrap.innerHTML = "";
        wrap.appendChild(el("div", { class: "empty", text: data.note || "No live matches right now." }));
        return;
      }
      wrap.innerHTML = "";
      const list = el("div", { class: "match-list" });
      for (const m of data.matches) list.appendChild(renderMatchCard(m));
      wrap.appendChild(list);
    } catch (err) {
      wrap.className = "error";
      wrap.textContent = `Couldn't load live scores: ${err.message}`;
      stopLivePolling();
    }
  }

  // ---- Top scorers --------------------------------------------------

  async function loadScorers() {
    const wrap = $("#scorers-wrap");
    wrap.className = "loading";
    wrap.textContent = "Loading top scorers…";
    try {
      const data = await API.topScorers(25);
      wrap.className = "";
      wrap.innerHTML = "";
      if (!data.top_scorers.length) {
        wrap.appendChild(el("div", { class: "empty", text: "No goals scored yet." }));
        return;
      }
      const table = el("table", { class: "scorers" });
      table.appendChild(el("thead", {}, el("tr", {}, [
        el("th", { text: "#" }), el("th", { text: "Player" }), el("th", { text: "Team" }), el("th", { text: "Goals" }),
      ])));
      const tbody = el("tbody");
      data.top_scorers.forEach((row, i) => {
        tbody.appendChild(el("tr", {}, [
          el("td", { class: "rank", text: String(i + 1) }),
          el("td", { text: row.player }),
          el("td", { text: row.team }),
          el("td", { class: "goals", text: String(row.goals) }),
        ]));
      });
      table.appendChild(tbody);
      wrap.appendChild(table);
    } catch (err) {
      wrap.className = "error";
      wrap.textContent = `Couldn't load top scorers: ${err.message}`;
    }
  }

  // ---- Team detail ----------------------------------------------------

  async function loadTeam(name) {
    const wrap = $("#team-wrap");
    wrap.className = "loading";
    wrap.textContent = "Loading team…";
    try {
      const data = await API.team(name);
      wrap.className = "";
      wrap.innerHTML = "";

      wrap.appendChild(el("div", { class: "team-header" }, [
        el("h2", { text: data.team.name }),
        el("span", { class: "short-code", text: data.team.short_code }),
      ]));

      if (data.standing) {
        const s = data.standing;
        const stats = [
          ["Pos", s.position], ["Pld", s.played], ["W", s.won], ["D", s.drawn], ["L", s.lost],
          ["GF", s.goals_for], ["GA", s.goals_against], ["GD", s.goal_difference], ["Pts", s.points],
        ];
        const grid = el("div", { class: "stat-grid" });
        for (const [label, value] of stats) {
          grid.appendChild(el("div", { class: "stat-cell" }, [
            el("div", { class: "value", text: String(value) }),
            el("div", { class: "label", text: label }),
          ]));
        }
        wrap.appendChild(grid);
      }

      if (data.recent_form.length) {
        wrap.appendChild(el("h3", { class: "subheading", text: "Recent form" }));
        const strip = el("div", { class: "form-strip" });
        for (const f of data.recent_form.slice().reverse()) {
          strip.appendChild(el("div", { class: `form-pill ${f.result}`, text: f.result }));
        }
        wrap.appendChild(strip);
      }

      if (data.upcoming_fixtures.length) {
        wrap.appendChild(el("h3", { class: "subheading", text: "Upcoming fixtures" }));
        const list = el("div", { class: "match-list" });
        for (const m of data.upcoming_fixtures) list.appendChild(renderMatchCard(m));
        wrap.appendChild(list);
      }

      wrap.appendChild(el("h3", { class: "subheading", text: `Squad (${data.roster.length})` }));
      const grid = el("div", { class: "roster-grid" });
      const posOrder = { GK: 0, DEF: 1, MID: 2, FWD: 3 };
      const roster = data.roster.slice().sort((a, b) => posOrder[a.position] - posOrder[b.position] || a.squad_number - b.squad_number);
      for (const p of roster) {
        grid.appendChild(el("div", { class: "roster-card" }, [
          el("span", { class: "num", text: `#${p.squad_number}` }),
          el("span", { class: "name", text: p.name }),
          el("span", { class: "pos", text: p.position }),
        ]));
      }
      wrap.appendChild(grid);
    } catch (err) {
      wrap.className = "error";
      wrap.textContent = `Couldn't load team: ${err.message}`;
    }
  }

  // ---- Search -----------------------------------------------------------

  let searchDebounce = null;
  const searchInput = $("#search-input");
  const searchResults = $("#search-results");

  searchInput.addEventListener("input", () => {
    clearTimeout(searchDebounce);
    const q = searchInput.value.trim();
    if (!q) {
      searchResults.classList.add("hidden");
      return;
    }
    searchDebounce = setTimeout(() => runSearch(q), 200);
  });

  document.addEventListener("click", (e) => {
    if (!e.target.closest(".search-box")) searchResults.classList.add("hidden");
  });

  async function runSearch(q) {
    try {
      const data = await API.search(q);
      renderSearchResults(data);
    } catch (err) {
      searchResults.innerHTML = "";
      searchResults.appendChild(el("div", { class: "search-empty", text: `Search failed: ${err.message}` }));
      searchResults.classList.remove("hidden");
    }
  }

  function renderSearchResults(data) {
    searchResults.innerHTML = "";
    const total = data.teams.length + data.players.length;
    if (total === 0) {
      searchResults.appendChild(el("div", { class: "search-empty", text: "No matches." }));
    } else {
      for (const t of data.teams.slice(0, 6)) {
        searchResults.appendChild(el("div", { class: "search-result-item", onclick: () => selectSearchResult(t.name) }, [
          el("span", { text: t.name }),
          el("span", { class: "search-result-tag", text: "Team" }),
        ]));
      }
      for (const p of data.players.slice(0, 8)) {
        searchResults.appendChild(el("div", { class: "search-result-item", onclick: () => selectSearchResult(p.team) }, [
          el("span", { text: `${p.name} (${p.position})` }),
          el("span", { class: "search-result-tag", text: p.team }),
        ]));
      }
    }
    searchResults.classList.remove("hidden");
  }

  function selectSearchResult(teamName) {
    searchResults.classList.add("hidden");
    searchInput.value = "";
    showTeam(teamName);
  }

  // ---- Boot ---------------------------------------------------------

  showTab("table");
})();
