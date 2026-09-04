/**
 * The Studio: a shot list, takes rendered on the GPU box, the critics' scores, and the director's verdict.
 * Same buttons for the person, the chat, a WebMCP agent, and the ADK director (cortex/agent/director).
 */
import { useCallback, useEffect, useRef, useState } from "react";
import { api, errorMessage } from "../api";
import type { Character, CharacterStage, Scene, Shot, StudioBoard, Take } from "../api";
import { navigate } from "../lib/router";
import { useToast } from "../components/Toast";
import { EmptyState } from "../components/States";

const STATUS_LABEL: Record<Shot["status"], string> = { planned: "planned", rendering: "rendering", rendered: "rendered", approved: "approved", reshoot: "reshoot" };

export function StudioView({ refresh }: { refresh: number }) {
  const { toast } = useToast();
  const [board, setBoard] = useState<StudioBoard | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [logline, setLogline] = useState("");
  const [planning, setPlanning] = useState(false);
  const [sceneLogline, setSceneLogline] = useState("");
  const [scenePlanning, setScenePlanning] = useState<"filler" | "full" | null>(null);
  const [newChar, setNewChar] = useState<{ name: string; description: string; style: string } | null>(null);
  const [gpu, setGpu] = useState<{ ready: boolean; host: string | null; name?: string } | null>(null);
  const [tele, setTele] = useState<{ metrics: boolean; logs: boolean; mcp: boolean; grafana_url: string | null; metrics_sent: number } | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  const load = useCallback(async () => {
    try {
      const b = await api.studio.board();
      setBoard(b);
      setLogline((l) => l || b.logline || "");
      setErr(null);
    } catch (e) {
      setErr(errorMessage(e));
    }
  }, []);
  useEffect(() => {
    void load();
    api.lab.gpu().then((g) => setGpu({ ready: g.ready, host: g.host, name: g.gpu?.name })).catch(() => setGpu({ ready: false, host: null }));
    api.telemetry().then((t) => setTele({ metrics: t.metrics, logs: t.logs, mcp: t.mcp, grafana_url: t.grafana_url, metrics_sent: t.metrics_sent })).catch(() => undefined);
  }, [load, refresh]);
  // Poll while something renders or builds (a scene render drives its shots one at a time on the server).
  useEffect(() => {
    const busyShots = board?.shots.filter((s) => s.status === "rendering") ?? [];
    const busyChars = board?.characters.filter((c) => c.status === "building") ?? [];
    const busyScenes = board?.scenes.filter((sc) => sc.status === "rendering") ?? [];
    if (!busyShots.length && !busyChars.length && !busyScenes.length) return;
    const t = window.setInterval(async () => {
      for (const s of busyShots) {
        try {
          await api.studio.refresh(s.id);
        } catch {
          /* still running */
        }
      }
      for (const c of busyChars) {
        try {
          await api.studio.refreshCharacter(c.id);
        } catch {
          /* still building */
        }
      }
      void load();
    }, 5000);
    return () => window.clearInterval(t);
  }, [board, load]);

  const plan = async () => {
    if (!logline.trim()) return;
    setPlanning(true);
    try {
      await api.studio.plan(logline.trim(), 4);
      toast("Shots planned");
      await load();
    } catch (e) {
      toast(errorMessage(e), "error");
    } finally {
      setPlanning(false);
    }
  };
  const render = async (s: Shot, smoke?: boolean) => {
    try {
      const r = await api.studio.render(s.id, { smoke });
      toast(`Rendering ${s.title} on ${r.executor}`);
      await load();
      navigate({ kind: "lab", run: r.id });
    } catch (e) {
      toast(errorMessage(e), "error");
    }
  };
  const patch = async (s: Shot, p: Partial<Shot> & { director_note?: string }) => {
    try {
      await api.studio.update(s.id, p);
      await load();
    } catch (e) {
      toast(errorMessage(e), "error");
    }
  };
  const remove = async (s: Shot) => {
    if (!window.confirm(`Delete shot "${s.title}"?`)) return;
    await api.studio.remove(s.id);
    await load();
  };
  // characters
  const addCharacter = async () => {
    if (!newChar?.name.trim() || !newChar.description.trim()) return;
    try {
      await api.studio.addCharacter({ name: newChar.name.trim(), description: newChar.description.trim(), style: newChar.style.trim() });
      setNewChar(null);
      toast("Character added to the bible");
      await load();
    } catch (e) {
      toast(errorMessage(e), "error");
    }
  };
  const buildCharacter = async (c: Character, stage: CharacterStage) => {
    try {
      const r = await api.studio.buildCharacter(c.id, { stage });
      toast(`Building ${stage} of ${c.name} on ${r.executor}`);
      await load();
      navigate({ kind: "lab", run: r.id });
    } catch (e) {
      toast(errorMessage(e), "error");
    }
  };
  const patchCharacter = async (c: Character, p: Partial<Omit<Character, "builds" | "scores">>) => {
    try {
      await api.studio.updateCharacter(c.id, p);
      await load();
    } catch (e) {
      toast(errorMessage(e), "error");
    }
  };
  const removeCharacter = async (c: Character) => {
    if (!window.confirm(`Remove ${c.name} from the bible? (Files on the box stay.)`)) return;
    await api.studio.removeCharacter(c.id);
    await load();
  };
  // scenes
  const planScene = async (kind: "filler" | "full") => {
    const line = sceneLogline.trim() || logline.trim();
    if (!line) return;
    setScenePlanning(kind);
    try {
      const sc = await api.studio.planScene({ logline: line, kind });
      toast(`${kind === "filler" ? "Filler" : "Full"} scene planned: ${sc.shots.length} shots`);
      await load();
    } catch (e) {
      toast(errorMessage(e), "error");
    } finally {
      setScenePlanning(null);
    }
  };
  const renderScene = async (sc: Scene) => {
    try {
      await api.studio.renderScene(sc.id, { only_missing: false });
      toast(`Rendering ${sc.shots.length} shots of ${sc.title}, one at a time`);
      await load();
    } catch (e) {
      toast(errorMessage(e), "error");
    }
  };
  const assembleScene = async (sc: Scene) => {
    try {
      const r = await api.studio.assembleScene(sc.id);
      toast(`Assembled ${r.clips} clips${r.missing.length ? `; ${r.missing.length} shots have no clip yet` : ""}`);
      await load();
    } catch (e) {
      toast(errorMessage(e), "error");
    }
  };
  const removeScene = async (sc: Scene) => {
    if (!window.confirm(`Delete scene "${sc.title}"? Its shots stay on the board.`)) return;
    await api.studio.removeScene(sc.id, false);
    await load();
  };
  const upload = async (f: File) => {
    try {
      const r = await api.studio.upload(f);
      toast(`Keyframe ${r.name} added`);
      await load();
    } catch (e) {
      toast(errorMessage(e), "error");
    }
  };

  if (err) return <EmptyState title="Could not load the studio" hint={err} />;
  if (!board) return <EmptyState title="Loading the studio" />;
  return (
    <div className="studio">
      <div className="studio-head">
        <div className="studio-logline">
          <label className="muted small">Logline</label>
          <textarea value={logline} onChange={(e) => setLogline(e.target.value)} rows={2} placeholder="A kaiju of black coral rises from a storm and, for one moment, hesitates." spellCheck={false} />
          <div className="row">
            <button className="primary" onClick={() => void plan()} disabled={planning || !logline.trim()}>
              {planning ? "Planning…" : "Plan 4 shots"}
            </button>
            <button className="btn" onClick={() => api.studio.logline(logline).then(() => toast("Logline saved"))} disabled={!logline.trim()}>
              Save logline
            </button>
            <span className="grow" />
            <span className={`pill ${gpu?.ready ? "ok" : "bad"}`} title="Where takes render">
              {gpu?.ready ? `render farm · ${gpu.name ?? gpu.host}` : "render farm offline · smoke brick only"}
            </span>
            <span className={`pill ${tele?.metrics ? "ok" : ""}`} title="Take metrics pushed to Grafana Cloud; the director queries them through the Grafana MCP">
              {tele?.metrics ? `Grafana · ${tele.metrics_sent} points` : "Grafana off"}
            </span>
          </div>
        </div>
        <div className="studio-assets">
          <label className="muted small">Keyframes</label>
          <div className="asset-row">
            {board.assets.map((a) => (
              <img key={a} src={api.studio.assetUrl(a)} alt={a} title={a} />
            ))}
            <button className="btn sm" onClick={() => fileRef.current?.click()}>
              Add image
            </button>
            <input ref={fileRef} type="file" accept="image/*" hidden onChange={(e) => e.target.files?.[0] && void upload(e.target.files[0])} />
          </div>
          <span className="muted small">Or use a path on the box, e.g. ~/celwright_v3b/hero_v3.png</span>
        </div>
      </div>

      <section className="studio-section">
        <header className="section-head">
          <h3>Characters</h3>
          <span className="muted small">the bible: a description is the identity text; the hero image, hero set (the prototype) and LoRA are built on the box</span>
          <span className="grow" />
          <button className="btn sm" onClick={() => setNewChar(newChar ? null : { name: "", description: "", style: "cel shaded anime, flat bold colors, thick lineart" })}>
            {newChar ? "Cancel" : "New character"}
          </button>
        </header>
        {newChar && (
          <div className="char-new">
            <input value={newChar.name} onChange={(e) => setNewChar({ ...newChar, name: e.target.value })} placeholder="Name (Okuun)" aria-label="Character name" />
            <textarea value={newChar.description} onChange={(e) => setNewChar({ ...newChar, description: e.target.value })} rows={2} placeholder="Identity text: colossal kaiju of black coral and volcanic glass, hexagonal basalt plates, glowing cyan chest orbs, four clawed forelimbs, no eyes" spellCheck={false} aria-label="Description" />
            <input value={newChar.style} onChange={(e) => setNewChar({ ...newChar, style: e.target.value })} placeholder="Style" aria-label="Style" />
            <button className="primary sm" onClick={() => void addCharacter()} disabled={!newChar.name.trim() || !newChar.description.trim()}>
              Add
            </button>
          </div>
        )}
        {board.characters.length === 0 ? (
          <p className="muted small">No characters yet. Add one, or ask the chat: 'build a character called Okuun, a kaiju of black coral'.</p>
        ) : (
          <div className="char-grid">
            {board.characters.map((c) => (
              <CharacterCard key={c.id} ch={c} gpuReady={!!gpu?.ready} onBuild={buildCharacter} onPatch={patchCharacter} onRemove={removeCharacter} />
            ))}
          </div>
        )}
      </section>

      <section className="studio-section">
        <header className="section-head">
          <h3>Scenes</h3>
          <span className="muted small">filler = 2 to 4 short cutaways for the edit; full = dialogue and continuity</span>
          <span className="grow" />
        </header>
        <div className="scene-plan">
          <input value={sceneLogline} onChange={(e) => setSceneLogline(e.target.value)} placeholder={logline.trim() ? "Scene logline (blank: use the logline above)" : "Scene logline: dawn on the drowned pier, the plates steam"} aria-label="Scene logline" />
          <button className="btn sm" onClick={() => void planScene("filler")} disabled={scenePlanning != null || !(sceneLogline.trim() || logline.trim())}>
            {scenePlanning === "filler" ? "Planning…" : "Plan filler scene"}
          </button>
          <button className="btn sm" onClick={() => void planScene("full")} disabled={scenePlanning != null || !(sceneLogline.trim() || logline.trim())}>
            {scenePlanning === "full" ? "Planning…" : "Plan full scene"}
          </button>
        </div>
        {board.scenes.length === 0 ? (
          <p className="muted small">No scenes yet.</p>
        ) : (
          <div className="scene-grid">
            {board.scenes.map((sc) => (
              <SceneCard key={sc.id} scene={sc} characters={board.characters} gpuReady={!!gpu?.ready} onRender={renderScene} onAssemble={assembleScene} onRemove={removeScene} />
            ))}
          </div>
        )}
      </section>

      <section className="studio-section">
        <header className="section-head">
          <h3>Shots</h3>
          <span className="muted small">{board.shots.length} shots · {board.counts.rendered ?? 0} rendered · {board.counts.reshoot ?? 0} to reshoot</span>
        </header>
        {board.shots.length === 0 ? (
          <EmptyState title="No shots yet" hint="Write a logline and press Plan, or ask the chat: 'plan four shots for …'." />
        ) : (
          <div className="shot-grid">
            {board.shots.map((s) => (
              <ShotCard key={s.id} shot={s} assets={board.assets} characters={board.characters} gpuReady={!!gpu?.ready} onRender={render} onPatch={patch} onRemove={remove} />
            ))}
          </div>
        )}
      </section>
    </div>
  );
}

const CHAR_STATUS: Record<Character["status"], string> = { draft: "draft", building: "building", hero: "hero", heroset: "hero set", lora: "LoRA", failed: "failed" };

function CharacterCard({ ch, gpuReady, onBuild, onPatch, onRemove }: { ch: Character; gpuReady: boolean; onBuild: (c: Character, stage: CharacterStage) => void; onPatch: (c: Character, p: Partial<Omit<Character, "builds" | "scores">>) => void; onRemove: (c: Character) => void }) {
  const [desc, setDesc] = useState(ch.description);
  useEffect(() => setDesc(ch.description), [ch.description]);
  const busy = ch.status === "building";
  const sc = ch.scores ?? {};
  const last = ch.builds[ch.builds.length - 1];
  return (
    <article className={`shot char st-${ch.status}`}>
      <header>
        <span className="shot-id">{ch.id}</span>
        <input className="shot-title" value={ch.name} onChange={(e) => onPatch(ch, { name: e.target.value })} aria-label="Character name" />
        <span className={`pill st-${ch.status}`}>{CHAR_STATUS[ch.status]}</span>
        <button className="icon-btn" onClick={() => onRemove(ch)} title="Remove character" aria-label="Remove character">
          ×
        </button>
      </header>
      <div className="shot-body">
        <div className="shot-kf">
          {ch.hero_url ? <img src={ch.hero_url} alt={`hero of ${ch.name}`} /> : <div className="kf-empty">{ch.hero ? "hero on the box" : "no hero yet"}</div>}
          {ch.contact && <img src={ch.contact} alt="hero set contact sheet" className="contact" />}
        </div>
        <div className="shot-fields">
          <textarea value={desc} onChange={(e) => setDesc(e.target.value)} onBlur={() => desc !== ch.description && onPatch(ch, { description: desc })} rows={3} spellCheck={false} aria-label="Identity text" />
          <div className="char-paths muted small">
            <span title="hero image (asset name or a path on the box)">hero: {ch.hero ?? "—"}</span>
            <span title="hero set folder on the box (the identity prototype for --proto)">set: {ch.heroset_dir ?? "—"}</span>
            <span title="LoRA folder on the box">lora: {ch.lora_dir ?? "—"}</span>
          </div>
          {sc.proto_mean != null && (
            <p className="small">
              prototype {sc.proto_mean.toFixed(3)} (min {(sc.proto_min ?? 0).toFixed(3)}) · p_own {(sc.p_own ?? 0).toFixed(2)} · {sc.n_kept ?? 0} kept
            </p>
          )}
        </div>
      </div>
      <footer>
        <button className="primary sm" onClick={() => onBuild(ch, "hero")} disabled={busy} title={gpuReady ? "Generate the hero image from the description on the box" : "No GPU box: runs the smoke brick"}>
          Build hero
        </button>
        <button className="btn sm" onClick={() => onBuild(ch, "heroset")} disabled={busy} title="Framings of the hero, filtered by the identity critic; the mean embedding is the prototype">
          Build hero set
        </button>
        <button className="btn sm" onClick={() => onBuild(ch, "lora")} disabled={busy} title="Rank-16 UNet LoRA on the hero set">
          Train LoRA
        </button>
        <span className="grow" />
        {last ? (
          <button className={`take ${last.status === "done" ? "keep" : last.status}`} onClick={() => navigate({ kind: "lab", run: last.id })} title={`${last.stage ?? "build"} · ${last.status}${last.elapsed_s != null ? ` · ${Math.round(last.elapsed_s)} s` : ""}`}>
            {last.stage ?? "build"} {ch.builds.length > 1 ? `×${ch.builds.length}` : ""}
          </button>
        ) : (
          <span className="muted small">no builds yet</span>
        )}
      </footer>
    </article>
  );
}

function SceneCard({ scene, characters, gpuReady, onRender, onAssemble, onRemove }: { scene: Scene; characters: Character[]; gpuReady: boolean; onRender: (sc: Scene) => void; onAssemble: (sc: Scene) => void; onRemove: (sc: Scene) => void }) {
  const names = scene.characters.map((id) => characters.find((c) => c.id === id)?.name ?? id);
  const rendering = scene.status === "rendering";
  const anyClip = scene.shot_rows.some((r) => r.verdict);
  return (
    <article className={`shot scene st-${scene.status}`}>
      <header>
        <span className="shot-id">{scene.id.split("-")[0]}</span>
        <span className="shot-title">{scene.title}</span>
        <span className={`pill kind-${scene.kind}`}>{scene.kind}</span>
        <span className={`pill st-${scene.status}`}>{scene.status}</span>
        <button className="icon-btn" onClick={() => onRemove(scene)} title="Delete scene" aria-label="Delete scene">
          ×
        </button>
      </header>
      <div className="scene-meta muted small">
        <span>{scene.set?.name || "no set"}{scene.set?.splat ? ` · ${scene.set.splat}` : ""}</span>
        <span>{names.length ? names.join(", ") : "no characters"}</span>
        <span>{scene.shots.length} shots · {scene.duration_s.toFixed(1)} s</span>
      </div>
      <div className="scene-shots">
        {scene.shot_rows.map((r, i) => (
          <span key={r.id} className={`take ${r.verdict ?? (r.status === "rendering" ? "running" : r.status === "missing" ? "failed" : "")}`} title={`${r.id} · ${r.status}${r.verdict ? ` · ${r.verdict}` : ""}${r.frames ? ` · ${r.frames}f` : ""}`}>
            {i + 1}. {r.title}
          </span>
        ))}
      </div>
      {scene.dialogue?.length > 0 && (
        <div className="scene-dialogue small">
          {scene.dialogue.slice(0, 6).map((d, i) => (
            <div key={i}>
              <b>{d.who}</b>: {d.line}
            </div>
          ))}
        </div>
      )}
      {scene.continuity && <p className="small director-note">Continuity: {scene.continuity}</p>}
      {scene.strip && <img src={`${scene.strip}?t=${scene.status}`} alt="scene contact strip" className="contact" />}
      {scene.video && <video src={scene.video} controls loop muted playsInline />}
      <footer>
        <button className="primary sm" onClick={() => onRender(scene)} disabled={rendering || !scene.shots.length} title={gpuReady ? "Render each shot in order on the GPU box, one take at a time" : "No GPU box: smoke takes"}>
          {rendering ? "Rendering…" : "Render all"}
        </button>
        <button className="btn sm" onClick={() => onAssemble(scene)} disabled={rendering || !anyClip} title="Concatenate the latest kept take of each shot with ffmpeg">
          Assemble
        </button>
        {scene.video && (
          <a className="btn sm" href={scene.video} download={`${scene.id}.mp4`}>
            Download
          </a>
        )}
      </footer>
    </article>
  );
}

function ShotCard({ shot, assets, characters, gpuReady, onRender, onPatch, onRemove }: { shot: Shot; assets: string[]; characters: Character[]; gpuReady: boolean; onRender: (s: Shot, smoke?: boolean) => void; onPatch: (s: Shot, p: Partial<Shot> & { director_note?: string }) => void; onRemove: (s: Shot) => void }) {
  const [prompt, setPrompt] = useState(shot.prompt);
  const [kf, setKf] = useState(shot.keyframe ?? "");
  useEffect(() => {
    setPrompt(shot.prompt);
    setKf(shot.keyframe ?? "");
  }, [shot.prompt, shot.keyframe]);
  const last = shot.takes[shot.takes.length - 1];
  const kfUrl = shot.keyframe && !shot.keyframe.startsWith("~") && !shot.keyframe.startsWith("/") ? api.studio.assetUrl(shot.keyframe) : null;
  const ch = shot.character ? characters.find((c) => c.id === shot.character) : undefined;
  const canRender = !!(shot.keyframe || ch?.hero);
  return (
    <article className={`shot st-${shot.status}`}>
      <header>
        <span className="shot-id">{shot.id.split("-")[0]}</span>
        <input className="shot-title" value={shot.title} onChange={(e) => onPatch(shot, { title: e.target.value })} />
        <span className={`pill st-${shot.status}`}>{STATUS_LABEL[shot.status]}</span>
        <button className="icon-btn" onClick={() => onRemove(shot)} title="Delete shot" aria-label="Delete shot">
          ×
        </button>
      </header>
      <div className="shot-body">
        <div className="shot-kf">
          {last?.contact ? <img src={last.contact} alt="contact sheet of the latest take" className="contact" /> : kfUrl ? <img src={kfUrl} alt="keyframe" /> : ch?.hero_url ? <img src={ch.hero_url} alt={`hero of ${ch.name}`} /> : <div className="kf-empty">{ch?.hero ? `${ch.name}'s hero` : "no keyframe"}</div>}
          {last?.clip && (
            <video src={last.clip} controls loop muted playsInline />
          )}
        </div>
        <div className="shot-fields">
          <textarea value={prompt} onChange={(e) => setPrompt(e.target.value)} onBlur={() => prompt !== shot.prompt && onPatch(shot, { prompt })} rows={4} spellCheck={false} aria-label="Prompt" />
          <div className="row">
            <select className="select sm" value={kf} onChange={(e) => { setKf(e.target.value); onPatch(shot, { keyframe: e.target.value || null }); }} aria-label="Keyframe">
              <option value="">keyframe: none (smoke)</option>
              {assets.map((a) => (
                <option key={a} value={a}>
                  {a}
                </option>
              ))}
              <option value="~/celwright_v3b/hero_v3.png">~/celwright_v3b/hero_v3.png (on the box)</option>
              {kf && !assets.includes(kf) && !kf.startsWith("~/celwright_v3b/hero_v3") && <option value={kf}>{kf}</option>}
            </select>
            <select className="select sm" value={shot.character ?? ""} onChange={(e) => onPatch(shot, { character: e.target.value || "" })} aria-label="Character" title="Without a keyframe, the character's hero image is the key and its hero set the identity prototype">
              <option value="">character: none</option>
              {characters.map((c) => (
                <option key={c.id} value={c.id}>
                  {c.name}
                </option>
              ))}
            </select>
            <select className="select sm" value={String(shot.frames ?? 49)} onChange={(e) => onPatch(shot, { frames: Number(e.target.value) })} aria-label="Frames">
              {[17, 33, 49, 81].map((n) => (
                <option key={n} value={n}>
                  {n} frames
                </option>
              ))}
            </select>
            <select className="select sm" value={shot.size ?? "832x480"} onChange={(e) => onPatch(shot, { size: e.target.value })} aria-label="Size">
              {["832x480", "640x384", "480x832"].map((v) => (
                <option key={v} value={v}>
                  {v}
                </option>
              ))}
            </select>
          </div>
          {shot.notes && <p className="muted small">{shot.notes}</p>}
          {shot.director_note && <p className="small director-note">Director: {shot.director_note}</p>}
        </div>
      </div>
      <footer>
        <button className="primary sm" onClick={() => onRender(shot)} disabled={shot.status === "rendering"} title={gpuReady && canRender ? (shot.keyframe ? "Render on the GPU box" : `Render on the GPU box from ${ch?.name}'s hero`) : "No keyframe or character hero, or no GPU: runs the smoke brick"}>
          {shot.status === "rendering" ? "Rendering…" : gpuReady && canRender ? "Render on the 5090" : "Render (smoke)"}
        </button>
        {shot.status === "rendered" && (
          <button className="btn sm" onClick={() => onPatch(shot, { status: "approved" })}>
            Approve
          </button>
        )}
        {shot.status === "reshoot" && (
          <button className="btn sm" onClick={() => onRender(shot)}>
            Reshoot
          </button>
        )}
        <span className="grow" />
        <Takes takes={shot.takes} />
      </footer>
    </article>
  );
}

function Takes({ takes }: { takes: Take[] }) {
  if (!takes.length) return <span className="muted small">no takes yet</span>;
  return (
    <span className="takes">
      {takes.slice(-4).map((t, i) => (
        <button key={t.id} className={`take ${t.verdict ?? t.status}`} onClick={() => navigate({ kind: "lab", run: t.id })} title={`${t.status}${t.verdict ? ` · ${t.verdict}` : ""}${t.identity_mean != null ? ` · identity ${t.identity_mean.toFixed(2)} (min ${(t.identity_min ?? 0).toFixed(2)})` : ""}${t.flicker_mean != null ? ` · flicker ${t.flicker_mean.toFixed(2)}` : ""}${t.gen_s != null ? ` · ${Math.round(t.gen_s)} s` : ""}${t.origin && t.origin !== "ui" ? ` · by agent` : ""}`}>
          T{takes.length - Math.min(4, takes.length) + i + 1}
          {t.identity_mean != null ? ` ${t.identity_mean.toFixed(2)}` : ""}
        </button>
      ))}
    </span>
  );
}
