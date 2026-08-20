import { app } from "../../scripts/app.js";

const NODE_NAMES = new Set(["llama_cpp_agent_chat", "Llama-cpp Agent Chat"]);

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

app.registerExtension({
    name: "llama-cpp-vlm.agent-chat",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (!NODE_NAMES.has(nodeData.name) && !NODE_NAMES.has(nodeData.display_name)) {
            return;
        }

        const originalOnNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            originalOnNodeCreated?.apply(this, arguments);
            setReadOnly(this, "ai_question", true);
        };

        const originalOnExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            originalOnExecuted?.apply(this, arguments);
            const ui = message?.ui ?? message ?? {};
            const status = String(firstValue(ui.status) ?? "");
            const aiQuestion = String(firstValue(ui.ai_question) ?? "");

            setWidgetValue(this, "ai_question", aiQuestion);
            setReadOnly(this, "ai_question", true);

            if (status === "complete" || status === "awaiting_user") {
                setWidgetValue(this, "user_answer", "");
            }
        };
    },
});
