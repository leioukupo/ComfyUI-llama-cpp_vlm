import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NODE_NAMES = new Set(["llama_cpp_live_chat", "Llama-cpp Live Chat"]);
const ROUTE_PREFIX = "/llama_cpp_vlm/live_chat";

function getWidget(node, name) {
    return node.widgets?.find((widget) => widget.name === name);
}

function getWidgetInputs(widget) {
    const inputs = [];
    if (!widget) {
        return inputs;
    }
    for (const key of ["inputEl", "element", "el"]) {
        const item = widget[key];
        if (!item) {
            continue;
        }
        if (item.matches?.("textarea,input")) {
            inputs.push(item);
        }
        item.querySelectorAll?.("textarea,input").forEach((input) => inputs.push(input));
    }
    return inputs;
}

function setWidgetValue(node, name, value) {
    const widget = getWidget(node, name);
    if (!widget) {
        return;
    }
    widget.value = value ?? "";
    for (const input of getWidgetInputs(widget)) {
        input.value = widget.value;
    }
    node.setDirtyCanvas?.(true, true);
}

function setReadOnly(node, name, readOnly) {
    const widget = getWidget(node, name);
    if (!widget) {
        return;
    }
    widget.readonly = readOnly;
    widget.readOnly = readOnly;
    for (const input of getWidgetInputs(widget)) {
        input.readOnly = readOnly;
    }
}

function getNodeById(nodeId) {
    const id = String(nodeId ?? "");
    return app.graph?.getNodeById?.(id) || app.graph?._nodes_by_id?.[id] || null;
}

async function fetchSessionState(node) {
    if (!node?.id) {
        return null;
    }
    const response = await api.fetchApi(`${ROUTE_PREFIX}/state?node_id=${encodeURIComponent(node.id)}`);
    if (!response.ok) {
        return null;
    }
    const payload = await response.json();
    return payload?.ok ? payload.session : null;
}

function applySessionState(node, session) {
    if (!node || !session) {
        return;
    }
    const transcript = String(session.conversation ?? "");
    const status = String(session.status ?? "idle");
    setWidgetValue(node, "conversation_log", transcript);
    setWidgetValue(node, "live_status", status);
    setReadOnly(node, "conversation_log", true);
    setReadOnly(node, "live_status", true);
    updateLiveControls(node, session);
}

async function syncSessionState(node) {
    const session = await fetchSessionState(node);
    if (session) {
        applySessionState(node, session);
    }
}

async function postLiveAction(node, action) {
    if (!node?.id) {
        return null;
    }
    setLiveControlsBusy(node, true);
    const form = new FormData();
    form.append("node_id", node.id);
    const session = node.__llamaLiveSession;
    if (session?.state_uid) {
        form.append("state_uid", session.state_uid);
    }
    const message = getWidget(node, "user_message")?.value ?? "";
    if (action === "send") {
        form.append("message", message);
    }
    try {
        const response = await api.fetchApi(`${ROUTE_PREFIX}/${action}`, {
            method: "POST",
            body: form,
        });
        const payload = await response.json().catch(() => ({}));
        if (payload?.ok && payload.session) {
            node.__llamaLiveSession = payload.session;
            applySessionState(node, payload.session);
            if (action === "send") {
                setWidgetValue(node, "user_message", "");
            }
        } else {
            const error = payload?.error || response.statusText || "request failed";
            setWidgetValue(node, "live_status", `error: ${error}`);
            updateLiveControls(node, node.__llamaLiveSession, error);
        }
        return payload;
    } catch (error) {
        setWidgetValue(node, "live_status", `error: ${error?.message || error}`);
        updateLiveControls(node, node.__llamaLiveSession, error?.message || String(error));
        return { ok: false, error: error?.message || String(error) };
    } finally {
        setLiveControlsBusy(node, false);
    }
}

function makeButton(label, className, onClick) {
    const button = document.createElement("button");
    button.type = "button";
    button.textContent = label;
    button.className = className;
    button.style.flex = "1 1 0";
    button.style.height = "32px";
    button.style.border = "1px solid #666";
    button.style.borderRadius = "4px";
    button.style.background = "#2b2b2b";
    button.style.color = "#eee";
    button.style.cursor = "pointer";
    button.addEventListener("pointerdown", (event) => {
        event.preventDefault();
        event.stopPropagation();
    });
    button.addEventListener("mousedown", (event) => {
        event.preventDefault();
        event.stopPropagation();
    });
    button.addEventListener("click", (event) => {
        event.preventDefault();
        event.stopPropagation();
        onClick();
    });
    return button;
}

function installLiveControls(node) {
    if (node.__llamaLiveControls) {
        return;
    }

    if (typeof node.addDOMWidget === "function") {
        const container = document.createElement("div");
        container.className = "llama-cpp-vlm-live-controls";
        container.style.display = "flex";
        container.style.flexDirection = "column";
        container.style.gap = "6px";
        container.style.padding = "4px 0";
        container.style.width = "100%";
        container.addEventListener("pointerdown", (event) => event.stopPropagation());
        container.addEventListener("mousedown", (event) => event.stopPropagation());
        container.addEventListener("click", (event) => event.stopPropagation());

        const row = document.createElement("div");
        row.style.display = "flex";
        row.style.gap = "6px";
        row.style.width = "100%";

        const status = document.createElement("div");
        status.style.minHeight = "16px";
        status.style.fontSize = "11px";
        status.style.color = "#aaa";
        status.style.whiteSpace = "normal";

        const sendButton = makeButton("Send", "llama-cpp-vlm-live-send", () => {
            postLiveAction(node, "send").catch(() => {});
        });
        const endButton = makeButton("End", "llama-cpp-vlm-live-end", () => {
            postLiveAction(node, "end").catch(() => {});
        });

        row.append(sendButton, endButton);
        container.append(row, status);

        const widget = node.addDOMWidget("live_controls", "llama-live-controls", container, {
            serialize: false,
            hideOnZoom: false,
        });
        if (widget) {
            widget.serialize = false;
        }
        node.__llamaLiveControls = { container, sendButton, endButton, status };
        updateLiveControls(node, node.__llamaLiveSession);
        return;
    }

    if (!node.__llamaLiveButtonsAdded) {
        node.__llamaLiveButtonsAdded = true;
        node.addWidget("button", "Send", "Send", async () => {
            await postLiveAction(node, "send");
        });
        node.addWidget("button", "End", "End", async () => {
            await postLiveAction(node, "end");
        });
    }
}

function setLiveControlsBusy(node, busy) {
    const controls = node.__llamaLiveControls;
    if (!controls) {
        return;
    }
    if (!busy) {
        controls.sendButton.style.opacity = "1";
        controls.endButton.style.opacity = "1";
        updateLiveControls(node, node.__llamaLiveSession);
        return;
    }
    controls.sendButton.disabled = Boolean(busy);
    controls.endButton.disabled = Boolean(busy);
    controls.sendButton.style.opacity = busy ? "0.6" : "1";
    controls.endButton.style.opacity = busy ? "0.6" : "1";
}

function updateLiveControls(node, session, error = "") {
    const controls = node.__llamaLiveControls;
    if (!controls) {
        return;
    }
    const status = String(session?.status || "idle");
    controls.status.textContent = error ? `Error: ${error}` : `Status: ${status}`;
    const ended = ["ended", "error"].includes(status);
    controls.sendButton.disabled = ended;
    controls.endButton.disabled = ended;
    controls.sendButton.style.cursor = ended ? "not-allowed" : "pointer";
    controls.endButton.style.cursor = ended ? "not-allowed" : "pointer";
}

app.registerExtension({
    name: "llama-cpp-vlm.live-chat",
    setup() {
        api.addEventListener("llama_cpp_vlm.live_chat_state", (event) => {
            const session = event?.detail ?? event ?? {};
            const node = getNodeById(session.node_id);
            if (!node) {
                return;
            }
            node.__llamaLiveSession = session;
            applySessionState(node, session);
            if (session.event === "queued") {
                setWidgetValue(node, "user_message", "");
            }
        });
    },
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (!NODE_NAMES.has(nodeData.name) && !NODE_NAMES.has(nodeData.display_name)) {
            return;
        }

        const originalOnNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            originalOnNodeCreated?.apply(this, arguments);
            setReadOnly(this, "conversation_log", true);
            setReadOnly(this, "live_status", true);
            const conversationWidget = getWidget(this, "conversation_log");
            if (conversationWidget) {
                conversationWidget.serialize = false;
            }
            const statusWidget = getWidget(this, "live_status");
            if (statusWidget) {
                statusWidget.serialize = false;
            }

            installLiveControls(this);

            setTimeout(() => {
                syncSessionState(this).catch(() => {});
            }, 0);
        };

        const originalOnExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function () {
            originalOnExecuted?.apply(this, arguments);
            setTimeout(() => {
                syncSessionState(this).catch(() => {});
            }, 0);
        };
    },
});
