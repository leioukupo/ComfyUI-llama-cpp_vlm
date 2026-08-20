import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

const NODE_NAMES = new Set(["llama_cpp_live_chat", "Llama-cpp Live Chat"]);
const ROUTE_PREFIX = "/llama_cpp_vlm/live_chat";

function firstValue(value) {
    return Array.isArray(value) ? value[0] : value;
}

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
    }
    return payload;
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

            if (!this.__llamaLiveButtonsAdded) {
                this.__llamaLiveButtonsAdded = true;
                this.addWidget("button", "Send", "Send", async () => {
                    await postLiveAction(this, "send");
                });
                this.addWidget("button", "End", "End", async () => {
                    await postLiveAction(this, "end");
                });
            }

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
