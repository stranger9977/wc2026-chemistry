const FLAG_BASE = "assets/flags";
let _doc = null;
let _activeMetric = "xt";

async function loadDoc() {
  const res = await fetch("outputs/chemistry.json");
  return res.json();
}

function getJoi(pair, metric) {
  if (metric === "vaep" && pair.joi90_vaep != null) return pair.joi90_vaep;
  if (metric === "xg") return pair.joi90_xg ?? 0;
  return pair.joi90_xt ?? pair.joi90 ?? 0;
}

function posChip(pos) {
  if (!pos) return "";
  return `<span class="pos">${pos}</span>`;
}

function renderNationGrid(doc, metric) {
  const grid = document.getElementById("nation-grid");
  grid.innerHTML = "";
  const entries = Object.entries(doc.nations).sort((a, b) => {
    const ja = a[1].pairs[0] ? getJoi(a[1].pairs[0], metric) : 0;
    const jb = b[1].pairs[0] ? getJoi(b[1].pairs[0], metric) : 0;
    return jb - ja;
  });
  for (const [code, entry] of entries) {
    const top = entry.pairs[0];
    const flag = entry.squad.flag_iso;
    const card = document.createElement("a");
    card.className = "card";
    card.href = `nation.html?code=${code}&metric=${metric}`;
    const joiVal = top ? getJoi(top, metric) : null;
    const aPos = top ? posChip(top.player_a_position) : "";
    const bPos = top ? posChip(top.player_b_position) : "";
    card.innerHTML = `
      <h2><img class="flag" src="${FLAG_BASE}/${flag}.svg" alt=""/> ${entry.squad.nation}</h2>
      <div class="sub">${entry.squad.manager} · ${entry.squad.formation} · ${entry.coverage.matches} matches</div>
      <div class="pair">${top
        ? `${aPos}${top.player_a_display || top.player_a_name} + ${bPos}${top.player_b_display || top.player_b_name}`
        : "no in-squad pairs"}
        ${joiVal != null ? `<span class="joi"> · ${joiVal.toFixed(3)}</span>` : ""}</div>`;
    grid.appendChild(card);
  }
}

function renderLeaderboard(doc, metric) {
  const tbody = document.querySelector("#leaderboard tbody");
  tbody.innerHTML = "";
  // Sort leaderboard by chosen metric
  const rows = [...doc.leaderboard].sort((a, b) => getJoi(b, metric) - getJoi(a, metric));
  for (let i = 0; i < rows.length; i++) {
    const p = rows[i];
    const tr = document.createElement("tr");
    const flag = p.flag_iso
      ? `<img class="flag-inline" src="assets/flags/${p.flag_iso}.svg" alt="${p.nation_code || ''}"/> `
      : "";
    const aPos = posChip(p.player_a_position);
    const bPos = posChip(p.player_b_position);
    const joi = getJoi(p, metric);
    tr.innerHTML = `<td>${i + 1}</td>
      <td>${flag}${aPos}${p.player_a_display || p.player_a_name} + ${bPos}${p.player_b_display || p.player_b_name}</td>
      <td class="joi">${joi.toFixed(3)}</td>
      <td>${Math.round(p.minutes)}</td><td>${p.matches}</td>`;
    tbody.appendChild(tr);
  }
}

function setMetric(metric, doc) {
  _activeMetric = metric;
  document.querySelectorAll(".metric-btn").forEach(btn => {
    btn.classList.toggle("active", btn.dataset.metric === metric);
  });
  renderNationGrid(doc, metric);
  renderLeaderboard(doc, metric);
}

(async function () {
  _doc = await loadDoc();
  renderNationGrid(_doc, _activeMetric);
  renderLeaderboard(_doc, _activeMetric);

  // Wire metric toggle
  document.querySelectorAll(".metric-btn").forEach(btn => {
    btn.addEventListener("click", () => {
      const hasVaep = _doc.meta && _doc.meta.has_vaep;
      const hasXg = _doc.meta && _doc.meta.has_xg;
      if (btn.dataset.metric === "vaep" && !hasVaep) {
        btn.title = "VAEP model not yet computed";
        return;
      }
      if (btn.dataset.metric === "xg" && !hasXg) {
        btn.title = "xG-Chain not yet computed";
        return;
      }
      setMetric(btn.dataset.metric, _doc);
    });
  });

  // Disable VAEP button if no VAEP data
  const hasVaep = _doc.meta && _doc.meta.has_vaep;
  if (!hasVaep) {
    const vaepBtn = document.querySelector('.metric-btn[data-metric="vaep"]');
    if (vaepBtn) {
      vaepBtn.classList.add("disabled");
      vaepBtn.title = "VAEP model not yet computed";
    }
  }

  // Disable xG-Chain button if no xG data
  const hasXg = _doc.meta && _doc.meta.has_xg;
  if (!hasXg) {
    const xgBtn = document.querySelector('.metric-btn[data-metric="xg"]');
    if (xgBtn) {
      xgBtn.classList.add("disabled");
      xgBtn.title = "xG-Chain not yet computed";
    }
  }
})();
