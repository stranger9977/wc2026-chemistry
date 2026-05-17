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
  const doc = await (await fetch("outputs/chemistry.json")).json();
  const entry = doc.nations[code];
  if (!entry) {
    document.getElementById("title").textContent = `Unknown nation: ${code}`;
    return;
  }
  const squad = entry.squad;
  const players = entry.players_by_id || {};

  document.getElementById("title").textContent = squad.nation;
  document.getElementById("subtitle").textContent =
    `${squad.manager} · ${squad.formation} · ${entry.coverage.matches} matches in window · ${Object.keys(players).length} players`;

  const svg = d3.select("#pitch");

  // Pitch background
  svg.append("rect")
    .attr("x", 0).attr("y", 0).attr("width", 1050).attr("height", 680)
    .attr("fill", "#0e3b1f");
  // Outer boundary
  svg.append("rect").attr("x", 25).attr("y", 25).attr("width", 1000).attr("height", 630)
    .attr("fill", "none").attr("stroke", "#fff").attr("stroke-width", 2).attr("opacity", 0.7);
  // Halfway line
  svg.append("line").attr("x1", 525).attr("x2", 525).attr("y1", 25).attr("y2", 655)
    .attr("stroke", "#fff").attr("stroke-width", 2).attr("opacity", 0.7);
  // Centre circle
  svg.append("circle").attr("cx", 525).attr("cy", 340).attr("r", 80)
    .attr("fill", "none").attr("stroke", "#fff").attr("stroke-width", 2).attr("opacity", 0.7);
  // Penalty boxes
  svg.append("rect").attr("x", 25).attr("y", 200).attr("width", 165).attr("height", 280)
    .attr("fill", "none").attr("stroke", "#fff").attr("stroke-width", 2).attr("opacity", 0.5);
  svg.append("rect").attr("x", 860).attr("y", 200).attr("width", 165).attr("height", 280)
    .attr("fill", "none").attr("stroke", "#fff").attr("stroke-width", 2).attr("opacity", 0.5);

  // SPADL coords are 105 wide × 68 tall; viewBox is 1050 × 680 — scale by 10

  // Filter to pitch-relevant pairs only
  const pitchSet = new Set(entry.pitch_player_ids || Object.keys(entry.players_by_id).map(Number));
  const allEligible = entry.pairs.filter(p =>
    pitchSet.has(p.player_a_id) && pitchSet.has(p.player_b_id)
  );
  const TOP_N = 15;
  const sortedDesc = [...allEligible].sort((a, b) => b.joi90 - a.joi90);
  const topEdges = sortedDesc.slice(0, TOP_N);
  const dimEdges = sortedDesc.slice(TOP_N);

  const vmin = d3.min(topEdges, d => d.joi90) ?? 0;
  const vmax = d3.max(topEdges, d => d.joi90) ?? 0;

  // Dim edges first (low opacity, no tooltip clutter)
  svg.append("g").selectAll("line.dim")
    .data(dimEdges).enter().append("line")
      .attr("x1", d => (entry.players_by_id[d.player_a_id] ? entry.players_by_id[d.player_a_id].x * 10 : 0))
      .attr("y1", d => (entry.players_by_id[d.player_a_id] ? entry.players_by_id[d.player_a_id].y * 10 : 0))
      .attr("x2", d => (entry.players_by_id[d.player_b_id] ? entry.players_by_id[d.player_b_id].x * 10 : 0))
      .attr("y2", d => (entry.players_by_id[d.player_b_id] ? entry.players_by_id[d.player_b_id].y * 10 : 0))
      .attr("stroke", "#6b7280")
      .attr("stroke-width", 1)
      .attr("opacity", 0.12)
      .attr("stroke-linecap", "round");

  // Top edges
  svg.append("g").selectAll("line.top")
    .data(topEdges).enter().append("line")
      .attr("x1", d => entry.players_by_id[d.player_a_id].x * 10)
      .attr("y1", d => entry.players_by_id[d.player_a_id].y * 10)
      .attr("x2", d => entry.players_by_id[d.player_b_id].x * 10)
      .attr("y2", d => entry.players_by_id[d.player_b_id].y * 10)
      .attr("stroke", d => colorFor(d.joi90, vmin, vmax))
      .attr("stroke-width", d => 2.5 + 6 * Math.min(d.minutes, 600) / 600)
      .attr("stroke-linecap", "round")
      .attr("opacity", 0.95)
      .append("title")
        .text(d => `${d.player_a_display ?? d.player_a_name} + ${d.player_b_display ?? d.player_b_name}\nJOI90 ${d.joi90.toFixed(3)} · ${Math.round(d.minutes)} mins · ${d.matches} matches`);

  // Player markers — only for pitch_player_ids (top 11 by minutes)
  const pitchPlayers = (entry.pitch_player_ids || Object.keys(entry.players_by_id).map(Number));
  const placedPoints = [];
  function labelOffset(x, y) {
    for (const [px_, py_] of placedPoints) {
      if (Math.abs(px_ - x) < 60 && Math.abs(py_ - y) < 35) return 30;
    }
    return -20;
  }
  const g = svg.append("g");
  for (const pid of pitchPlayers) {
    const p = entry.players_by_id[pid];
    if (!p) continue;
    const x = p.x * 10, y = p.y * 10;
    g.append("circle").attr("cx", x).attr("cy", y).attr("r", 14)
      .attr("fill", "#fff").attr("stroke", "#000").attr("stroke-width", 1);
    g.append("circle").attr("cx", x).attr("cy", y).attr("r", 11)
      .attr("fill", squad.team_color);
    const dy = labelOffset(x, y);
    g.append("text").attr("x", x).attr("y", y + dy)
      .attr("text-anchor", "middle")
      .attr("fill", "#fff").attr("stroke", "#000").attr("stroke-width", 3).attr("paint-order", "stroke")
      .attr("font-size", 13).attr("font-weight", 700)
      .text(p.display_name);
    placedPoints.push([x, y]);
  }

  // Download buttons
  const dl = document.getElementById("downloads");
  for (const [label, file] of [
    ["Pitch SVG", "pitch.svg"], ["Pitch PNG 4K", "pitch.png"],
    ["Branded PNG", "pitch_branded.png"], ["Data JSON", "data.json"],
  ]) {
    const a = document.createElement("a");
    a.className = "btn"; a.href = `exports/nations/${code}/${file}`;
    a.textContent = `Download ${label}`; a.download = `${code}_${file}`;
    dl.appendChild(a);
  }

  // Pairs table — still shows all pairs for full squad context
  const tbody = document.querySelector("#pairs tbody");
  entry.pairs.forEach((p, i) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `<td>${i+1}</td><td>${p.player_a_display || p.player_a_name} + ${p.player_b_display || p.player_b_name}</td>
      <td class="joi">${p.joi90.toFixed(3)}</td><td>${Math.round(p.minutes)}</td><td>${p.matches}</td>`;
    tbody.appendChild(tr);
  });
}
main();
