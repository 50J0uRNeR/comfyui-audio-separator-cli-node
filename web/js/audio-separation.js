import { app } from "/scripts/app.js"

const NODE_TYPE = "AudioSeparation"

function widgetValue(node, name) {
    const w = node.widgets?.find((x) => x.name === name)
    return w ? String(w.value ?? "") : ""
}

function styleWidget(w, widthPct) {
    const c = w.container
    if (!c) return
    c.style.display = "inline-flex"
    c.style.width = `${widthPct}%`
    c.style.verticalAlign = "top"
    c.style.boxSizing = "border-box"
}

// command_options holds long free text, so it gets a full-width row of its own.
function styleOptionsRow(w) {
    const c = w.container
    if (!c) return
    c.style.display = "block"
    c.style.width = "100%"
    c.style.boxSizing = "border-box"
}

app.registerExtension({
    name: "AudioSeparatorCLI.layout",
    beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== NODE_TYPE) return
        const onAdded = nodeType.prototype.onAdded
        nodeType.prototype.onAdded = function () {
            onAdded?.apply(this, arguments)
            this.size[0] = Math.max(this.size[0], 560)
            for (const name of ["input_files", "output_dir", "model_filename"]) {
                const w = this.widgets.find((x) => x.name === name)
                if (w) styleWidget(w, 32)
            }
            const opts = this.widgets.find((x) => x.name === "command_options")
            if (opts) styleOptionsRow(opts)

            // Stacked full-width buttons (ComfyUI widget containers don't flow
            // side-by-side reliably): Show Command Help, List Loaded Models,
            // List Model Stems, Remove Models, STOP PROCESS.
            const helpBtn = this.addWidget("button", "Show Command Help", "", async () => {
                try {
                    const r = await fetch("/api/audio_separation/help")
                    let text = `HTTP ${r.status}`
                    try { text = (await r.json()).help || text } catch (_) {}
                    alert(text)
                } catch (e) {
                    alert("Show Command Help failed: " + e)
                }
            }, {})
            if (helpBtn.container) {
                helpBtn.container.style.width = "100%"
                helpBtn.container.style.textAlign = "left"
                helpBtn.container.style.marginTop = "4px"
            }
            const localBtn = this.addWidget("button", "List Loaded Models", "", async () => {
                try {
                    const r = await fetch("/api/audio_separation/local_models")
                    let names = [], dir = ""
                    try { ({ models: names, dir } = await r.json()); names = names || [] } catch (_) {}
                    if (!names.length) alert(`No models found in:\n${dir}`)
                    else alert(names.join("\n"))
                } catch (e) {
                    alert("List Loaded Models failed: " + e)
                }
            }, {})
            if (localBtn.container) {
                localBtn.container.style.width = "100%"
                localBtn.container.style.textAlign = "left"
                localBtn.container.style.marginTop = "4px"
            }
            let stemsBusy = false  // per-node re-entry guard (label updates don't render here)
            const stemsBtn = this.addWidget("button", "List Model Stems", "", async () => {
                if (stemsBusy) return  // a resolution is already in flight; ignore re-clicks
                const modelFile = widgetValue(this, "model_filename")
                if (!modelFile) { alert("Select a model first."); return }
                stemsBusy = true
                try {
                    // Only the model's ~KB yaml config is fetched (never its weights), so
                    // this resolves quickly; show the result in an alert.
                    const r = await fetch(
                        `/api/audio_separation/models?model_filename=${encodeURIComponent(modelFile)}`)
                    let data = {}
                    try { data = await r.json() } catch (_) {}
                    const stems = data.stems || []
                    if (!stems.length) alert(`${data.message || "No stem info"}\n(${modelFile})`)
                    // Number each stem 1..N so it lines up with the node's output_stempath_N slots.
                    else alert(stems.map((s, i) => `${i + 1}. ${s}`).join("\n"))
                } catch (e) {
                    alert("List Model Stems failed: " + e)
                } finally {
                    stemsBusy = false
                }
            }, {})
            if (stemsBtn.container) {
                stemsBtn.container.style.width = "100%"
                stemsBtn.container.style.textAlign = "left"
                stemsBtn.container.style.marginTop = "4px"
            }
            // Remove Models: destructive failsafe — wipes the entire models folder.
            const removeBtn = this.addWidget("button", "Remove Models", "", async () => {
                if (!confirm("Remove ALL downloaded models?\n\nThis deletes EVERY file in the models folder.\nThis cannot be undone. Continue?")) return
                try {
                    const r = await fetch("/api/audio_separation/remove_models", { method: "POST" })
                    let msg = `HTTP ${r.status}`
                    try { msg = (await r.json()).message || msg } catch (_) {}
                    alert(msg)
                } catch (e) {
                    alert("Remove Models failed: " + e)
                }
            }, {})
            if (removeBtn.container) {
                removeBtn.container.style.width = "100%"
                removeBtn.container.style.textAlign = "left"
                removeBtn.container.style.marginTop = "4px"
            }
            // STOP PROCESS: SIGKILLs a frozen audio-separator.
            const btn = this.addWidget("button", "STOP PROCESS", "", async () => {
                try {
                    const r = await fetch("/api/audio_separation/kill", { method: "POST" })
                    let msg = `HTTP ${r.status}`
                    try { msg = (await r.json()).message || msg } catch (_) {}
                    alert(msg)
                } catch (e) {
                    alert("STOP PROCESS failed: " + e)
                }
            }, {})
            if (btn.container) {
                btn.container.style.width = "100%"
                btn.container.style.textAlign = "right"
                btn.container.style.marginTop = "4px"
            }
        }
    },
})
