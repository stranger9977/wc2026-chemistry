const POS = {
  GK:  [80, 340], RB: [250, 580], RWB: [400, 600], RCB: [220, 440],
  CB:  [220, 340], LCB: [220, 240], LB: [250, 100], LWB: [400, 80],
  RDM: [350, 420], DM: [350, 340], LDM: [350, 260],
  RCM: [550, 420], CM: [550, 340], LCM: [550, 260],
  RM:  [600, 560], LM:  [600, 120],
  RAM: [700, 420], AM:  [700, 340], LAM: [700, 260],
  RW:  [850, 560], LW:  [850, 120], ST:  [900, 340], CF: [900, 340], SS: [820, 340],
};

function lerpColor(c, h, t) {
  const a = c.match(/.{2}/g).map(s => parseInt(s, 16));
  const b = h.match(/.{2}/g).map(s => parseInt(s, 16));
  return "#" + a.map((v, i) => Math.round(v + (b[i] - v) * t).toString(16).padStart(2, "0")).join("");
}

function colorFor(joi, min, max) {
  if (max === min) return "#4ade80";
  const t = Math.max(0, Math.min(1, (joi - min) / (max - min)));
  return lerpColor("ef4444", "4ade80", t);
}

async function main() {
  const params = new URLSearchParams(location.search);
  const code = params.get("code");
  const doc = await (await fetch("../outputs/chemistry.json")).json();
  const entry = doc.nations[code];
  if (!entry) {
    document.getElementById("title").textContent = `Unknown nation: ${code}`;
    return;
  }
  const squad = entry.squad;

  document.getElementById("title").textContent = squad.nation;
  document.getElementById("subtitle").textContent =
    `${squad.manager} · ${squad.formation} · ${entry.coverage.matches} matches in window`;

  const svg = d3.select("#pitch");
  svg.append("rect")
    .attr("x", 0).attr("y", 0).attr("width", 1050).attr("height", 680)
    .attr("fill", "#1a4d2e").attr("opacity", 0.4);

  svg.append("rect").attr("x", 10).attr("y", 10).attr("width", 1030).attr("height", 660)
    .attr("fill", "none").attr("stroke", "#fff").attr("stroke-width", 2);
  svg.append("line").attr("x1", 525).attr("x2", 525).attr("y1", 10).attr("y2", 670)
    .attr("stroke", "#fff").attr("stroke-width", 2);
  svg.append("circle").attr("cx", 525).attr("cy", 340).attr("r", 80)
    .attr("fill", "none").attr("stroke", "#fff").attr("stroke-width", 2);

  const namePos = {};
  for (const p of squad.players) {
    namePos[p.name] = POS[p.position] || [525, 340];
  }

  const pairs = entry.pairs;
  const vmin = d3.min(pairs, d => d.joi90) ?? 0;
  const vmax = d3.max(pairs, d => d.joi90) ?? 0;

  svg.append("g").selectAll("line")
    .data(pairs).enter().append("line")
      .attr("x1", d => (namePos[d.player_a_name] || [0, 0])[0])
      .attr("y1", d => (namePos[d.player_a_name] || [0, 0])[1])
      .attr("x2", d => (namePos[d.player_b_name] || [0, 0])[0])
      .attr("y2", d => (namePos[d.player_b_name] || [0, 0])[1])
      .attr("stroke", d => colorFor(d.joi90, vmin, vmax))
      .attr("stroke-width", d => 2 + 6 * Math.min(d.minutes, 600) / 600)
      .attr("stroke-linecap", "round")
      .attr("opacity", 0.85)
      .append("title")
        .text(d => `${d.player_a_name} + ${d.player_b_name}\nJOI90 ${d.joi90.toFixed(3)} · ${Math.round(d.minutes)} mins · ${d.matches} matches`);

  const g = svg.append("g");
  for (const [name, [x, y]] of Object.entries(namePos)) {
    g.append("circle").attr("cx", x).attr("cy", y).attr("r", 14)
      .attr("fill", "#fff");
    g.append("circle").attr("cx", x).attr("cy", y).attr("r", 11)
      .attr("fill", squad.team_color);
    g.append("text").attr("x", x).attr("y", y - 18)
      .attr("text-anchor", "middle").attr("fill", "#fff")
      .attr("font-size", 14).attr("font-weight", 700).text(name);
  }

  const dl = document.getElementById("downloads");
  for (const [label, file] of [
    ["Pitch SVG", "pitch.svg"], ["Pitch PNG 4K", "pitch.png"],
    ["Branded PNG", "pitch_branded.png"], ["Data JSON", "data.json"],
  ]) {
    const a = document.createElement("a");
    a.className = "btn"; a.href = `../exports/nations/${code}/${file}`;
    a.textContent = `Download ${label}`; a.download = `${code}_${file}`;
    dl.appendChild(a);
  }

  const tbody = document.querySelector("#pairs tbody");
  pairs.forEach((p, i) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `<td>${i+1}</td><td>${p.player_a_name} + ${p.player_b_name}</td>
      <td class="joi">${p.joi90.toFixed(3)}</td><td>${Math.round(p.minutes)}</td><td>${p.matches}</td>`;
    tbody.appendChild(tr);
  });
}
main();
