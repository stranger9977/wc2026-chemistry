const FLAG_BASE = "assets/flags";

async function loadDoc() {
  const res = await fetch("outputs/chemistry.json");
  return res.json();
}

function renderNationGrid(doc) {
  const grid = document.getElementById("nation-grid");
  const entries = Object.entries(doc.nations).sort((a, b) => {
    const ja = a[1].pairs[0]?.joi90 ?? 0;
    const jb = b[1].pairs[0]?.joi90 ?? 0;
    return jb - ja;
  });
  for (const [code, entry] of entries) {
    const top = entry.pairs[0];
    const flag = entry.squad.flag_iso;
    const card = document.createElement("a");
    card.className = "card";
    card.href = `nation.html?code=${code}`;
    card.innerHTML = `
      <h2><img class="flag" src="${FLAG_BASE}/${flag}.svg" alt=""/> ${entry.squad.nation}</h2>
      <div class="sub">${entry.squad.manager} · ${entry.squad.formation} · ${entry.coverage.matches} matches</div>
      <div class="pair">${top ? `${top.player_a_name} + ${top.player_b_name}` : "no in-squad pairs"}
        ${top ? `<span class="joi"> · ${top.joi90.toFixed(3)}</span>` : ""}</div>`;
    grid.appendChild(card);
  }
}

function renderLeaderboard(doc) {
  const tbody = document.querySelector("#leaderboard tbody");
  for (let i = 0; i < doc.leaderboard.length; i++) {
    const p = doc.leaderboard[i];
    const tr = document.createElement("tr");
    tr.innerHTML = `<td>${i + 1}</td>
      <td>${p.player_a_name} + ${p.player_b_name}</td>
      <td class="joi">${p.joi90.toFixed(3)}</td>
      <td>${Math.round(p.minutes)}</td><td>${p.matches}</td>`;
    tbody.appendChild(tr);
  }
}

(async function () {
  const doc = await loadDoc();
  renderNationGrid(doc);
  renderLeaderboard(doc);
})();
