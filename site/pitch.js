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
  function px(p) { return [p.x * 10, p.y * 10]; }

  const pairs = entry.pairs;
  const vmin = d3.min(pairs, d => d.joi90) ?? 0;
  const vmax = d3.max(pairs, d => d.joi90) ?? 0;

  // Edges — look up by player_id (numeric key stored as string in JSON)
  svg.append("g").selectAll("line")
    .data(pairs.filter(p => players[p.player_a_id] && players[p.player_b_id]))
    .enter().append("line")
      .attr("x1", d => px(players[d.player_a_id])[0])
      .attr("y1", d => px(players[d.player_a_id])[1])
      .attr("x2", d => px(players[d.player_b_id])[0])
      .attr("y2", d => px(players[d.player_b_id])[1])
      .attr("stroke", d => colorFor(d.joi90, vmin, vmax))
      .attr("stroke-width", d => 2 + 6 * Math.min(d.minutes, 600) / 600)
      .attr("stroke-linecap", "round")
      .attr("opacity", 0.85)
      .append("title")
        .text(d => `${d.player_a_name} + ${d.player_b_name}\nJOI90 ${d.joi90.toFixed(3)} · ${Math.round(d.minutes)} mins · ${d.matches} matches`);

  // Player markers + smart label placement
  const placedPoints = [];
  function labelOffset(x, y) {
    for (const [px_, py_] of placedPoints) {
      if (Math.abs(px_ - x) < 60 && Math.abs(py_ - y) < 35) {
        return 30;  // push below the marker
      }
    }
    return -20;  // default: above the marker
  }

  const g = svg.append("g");
  for (const [idStr, p] of Object.entries(players)) {
    const [x, y] = px(p);
    g.append("circle").attr("cx", x).attr("cy", y).attr("r", 14)
      .attr("fill", "#fff").attr("stroke", "#000").attr("stroke-width", 1);
    g.append("circle").attr("cx", x).attr("cy", y).attr("r", 11)
      .attr("fill", squad.team_color);
    const dy = labelOffset(x, y);
    const txt = g.append("text").attr("x", x).attr("y", y + dy)
      .attr("text-anchor", "middle")
      .attr("fill", "#fff")
      .attr("stroke", "#000").attr("stroke-width", 3).attr("paint-order", "stroke")
      .attr("font-size", 13).attr("font-weight", 700).text(p.name);
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

  // Pairs table
  const tbody = document.querySelector("#pairs tbody");
  pairs.forEach((p, i) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `<td>${i+1}</td><td>${p.player_a_name} + ${p.player_b_name}</td>
      <td class="joi">${p.joi90.toFixed(3)}</td><td>${Math.round(p.minutes)}</td><td>${p.matches}</td>`;
    tbody.appendChild(tr);
  });
}
main();
