package openai

import (
	"encoding/json"
	"fmt"
	"strings"
	"testing"

	"github.com/nguyenha935/kiro-cpa-plugin/internal/modelcapabilities"
)

func TestBuildKiroPayloadPreservesInstructionsAndCompleteHistory(t *testing.T) {
	messages := make([]map[string]any, 0, 62)
	messages = append(messages, map[string]any{"role": "system", "content": "Keep answers short."})
	for i := 0; i < 30; i++ {
		messages = append(messages,
			map[string]any{"role": "user", "content": fmt.Sprintf("question %d", i)},
			map[string]any{"role": "assistant", "content": fmt.Sprintf("answer %d", i)},
		)
	}
	messages = append(messages, map[string]any{"role": "user", "content": "latest"})
	request, err := json.Marshal(map[string]any{"model": "claude-opus-5", "messages": messages})
	if err != nil {
		t.Fatal(err)
	}

	raw, _ := BuildKiroPayloadFromOpenAI(withDeclaredTestTools(t, request), "claude-opus-5", "profile", "AI_EDITOR", modelcapabilities.Capability{}, "")
	var payload KiroPayload
	if err := json.Unmarshal(raw, &payload); err != nil {
		t.Fatalf("invalid payload: %v", err)
	}
	if got := payload.ConversationState.History[0].UserInputMessage.Content; got != "Keep answers short.\n\nquestion 0" {
		t.Fatalf("instructions were not anchored to the first user message: %q", got)
	}
	if got := payload.ConversationState.CurrentMessage.UserInputMessage.Content; got != "latest" {
		t.Fatalf("current message was changed: %q", got)
	}
	if len(payload.ConversationState.History) != 60 {
		t.Fatalf("history length = %d, want all 60 messages", len(payload.ConversationState.History))
	}
	for i, message := range payload.ConversationState.History {
		if i%2 == 0 && message.UserInputMessage == nil {
			t.Fatalf("history[%d] must be a user message", i)
		}
		if i%2 == 1 && message.AssistantResponseMessage == nil {
			t.Fatalf("history[%d] must be an assistant message", i)
		}
	}
}

func TestBuildKiroPayloadAnchorsSystemAndDeveloperBeforeToolContinuation(t *testing.T) {
	t.Parallel()

	request := []byte(`{
		"messages": [
			{"role":"system","content":"Base rules."},
			{"role":"developer","content":"Repository rules."},
			{"role":"user","content":"Run pwd"},
			{"role":"assistant","content":null,"tool_calls":[{"id":"call_1","type":"function","function":{"name":"exec_command","arguments":"{\"cmd\":\"pwd\"}"}}]},
			{"role":"tool","tool_call_id":"call_1","content":"ok"}
		]
	}`)

	raw, _ := BuildKiroPayloadFromOpenAI(withDeclaredTestTools(t, request), "claude-opus-5", "profile", "AI_EDITOR", modelcapabilities.Capability{}, "")
	var payload KiroPayload
	if err := json.Unmarshal(raw, &payload); err != nil {
		t.Fatalf("invalid payload: %v", err)
	}
	if got := payload.ConversationState.History[0].UserInputMessage.Content; got != "Base rules.\nRepository rules.\n\nRun pwd" {
		t.Fatalf("first user content = %q", got)
	}
	current := payload.ConversationState.CurrentMessage.UserInputMessage
	if current.Content != "" {
		t.Fatalf("tool-result current content = %q, want empty", current.Content)
	}
	if current.UserInputMessageContext == nil || len(current.UserInputMessageContext.ToolResults) != 1 {
		t.Fatalf("current tool results = %#v", current.UserInputMessageContext)
	}
}

func TestBuildKiroPayloadPreservesDeveloperInstructions(t *testing.T) {
	request := []byte(`{"messages":[{"role":"developer","content":"Use the repository conventions."},{"role":"user","content":"Fix the test."}]}`)

	raw, _ := BuildKiroPayloadFromOpenAI(request, "claude-opus-5", "profile", "AI_EDITOR", modelcapabilities.Capability{}, "")
	var payload KiroPayload
	if err := json.Unmarshal(raw, &payload); err != nil {
		t.Fatalf("invalid payload: %v", err)
	}
	if got := payload.ConversationState.CurrentMessage.UserInputMessage.Content; got != "Use the repository conventions.\n\nFix the test." {
		t.Fatalf("developer instructions or current message were not mapped: %q", got)
	}
}

func TestBuildKiroPayloadFlattensHistoryForUndeclaredTool(t *testing.T) {
	request := []byte(`{
		"messages":[
			{"role":"user","content":"Read the file."},
			{"role":"assistant","content":null,"tool_calls":[{"id":"call_old","type":"function","function":{"name":"OldTool","arguments":"{\"path\":\"README.md\"}"}}]},
			{"role":"tool","tool_call_id":"call_old","content":"contents"}
		],
		"tools":[{"type":"function","function":{"name":"NewTool","description":"A replacement tool","parameters":{"type":"object"}}}]
	}`)

	raw, _ := BuildKiroPayloadFromOpenAI(request, "claude-opus-5", "profile", "AI_EDITOR", modelcapabilities.Capability{}, "")
	var payload KiroPayload
	if err := json.Unmarshal(raw, &payload); err != nil {
		t.Fatalf("invalid payload: %v", err)
	}
	assistant := payload.ConversationState.History[1].AssistantResponseMessage
	if len(assistant.ToolUses) != 0 || !strings.Contains(assistant.Content, `Historical unavailable tool call (not an executable call):`) || !strings.Contains(assistant.Content, `"toolUseId":"call_old"`) {
		t.Fatalf("undeclared tool use was not flattened: %#v", assistant)
	}
	current := payload.ConversationState.CurrentMessage.UserInputMessage
	if current.UserInputMessageContext == nil || len(current.UserInputMessageContext.ToolResults) != 0 {
		t.Fatalf("structured tool results were retained: %#v", current.UserInputMessageContext)
	}
	if !strings.Contains(current.Content, `Historical unavailable tool result:`) || !strings.Contains(current.Content, `"toolUseId":"call_old"`) || !strings.Contains(current.Content, "contents") {
		t.Fatalf("tool result text was not preserved: %q", current.Content)
	}
}

func TestBuildKiroPayloadKeepsDeclaredToolsWithUnknownHistory(t *testing.T) {
	for _, historicalResults := range []bool{false, true} {
		t.Run(map[bool]string{false: "current_mixed_results", true: "historical_mixed_results"}[historicalResults], func(t *testing.T) {
			suffix := ""
			if historicalResults {
				suffix = `,
				{"role":"assistant","content":"  keep these bytes\n","tool_calls":[{"id":"call_next","type":"function","function":{"name":"Bash","arguments":"{\"command\":\"kubectl version --client\"}"}}]},
				{"role":"tool","tool_call_id":"call_next","content":"next result"}`
			}
			request := []byte(`{
				"messages":[
					{"role":"user","content":"Inspect the cluster."},
					{"role":"assistant","content":null,"tool_calls":[
						{"id":"call_known","type":"function","function":{"name":"Bash","arguments":"{\"command\":\"kubectl get pods\"}"}},
						{"id":"call_unknown","type":"function","function":{"name":"Grep","arguments":"{\"pattern\":\"cliproxy\"}"}}
					]},
					{"role":"tool","tool_call_id":"call_known","content":"known result"},
					{"role":"tool","tool_call_id":"call_unknown","content":"No such tool available: Grep"}` + suffix + `
				],
				"tools":[{"type":"function","function":{"name":"Bash","description":"Run a command","parameters":{"type":"object"}}}]
			}`)
			raw, _ := BuildKiroPayloadFromOpenAI(request, "claude-opus-5.5", "profile", "AI_EDITOR", modelcapabilities.Capability{}, "")
			var payload KiroPayload
			if err := json.Unmarshal(raw, &payload); err != nil {
				t.Fatal(err)
			}
			history := payload.ConversationState.History
			assistant := history[1].AssistantResponseMessage
			if len(assistant.ToolUses) != 1 || assistant.ToolUses[0].ToolUseID != "call_known" || assistant.ToolUses[0].Input["command"] != "kubectl get pods" {
				t.Fatalf("declared call changed: %#v", assistant.ToolUses)
			}
			current := payload.ConversationState.CurrentMessage.UserInputMessage
			mixed := current
			if historicalResults {
				mixed = *history[2].UserInputMessage
				if history[3].AssistantResponseMessage.Content != "  keep these bytes\n" {
					t.Fatalf("unaffected assistant text changed: %q", history[3].AssistantResponseMessage.Content)
				}
				if len(current.UserInputMessageContext.ToolResults) != 1 || current.UserInputMessageContext.ToolResults[0].ToolUseID != "call_next" || current.Content != "" {
					t.Fatalf("current results changed: %#v", current)
				}
			}
			if len(mixed.UserInputMessageContext.ToolResults) != 1 || mixed.UserInputMessageContext.ToolResults[0].ToolUseID != "call_known" || mixed.UserInputMessageContext.ToolResults[0].Content[0].Text != "known result" {
				t.Fatalf("declared result changed: %#v", mixed)
			}
			if !strings.Contains(mixed.Content, `"toolUseId":"call_unknown"`) || !strings.Contains(mixed.Content, "No such tool available: Grep") {
				t.Fatalf("unknown result lost: %q", mixed.Content)
			}
			if strings.Contains(assistant.Content+mixed.Content, "<tool_use") || strings.Contains(assistant.Content+mixed.Content, "<tool_result") || strings.Contains(assistant.Content, `"name":"Bash"`) {
				t.Fatalf("tool protocol leaked into text: %q %q", assistant.Content, mixed.Content)
			}
		})
	}
}

func TestBuildKiroPayloadKeepsHistoryForDeclaredTool(t *testing.T) {
	request := []byte(`{
		"messages":[
			{"role":"user","content":"Read the file."},
			{"role":"assistant","content":null,"tool_calls":[{"id":"call_1","type":"function","function":{"name":"Read","arguments":"{\"path\":\"README.md\"}"}}]},
			{"role":"tool","tool_call_id":"call_1","content":"contents"}
		],
		"tools":[{"type":"function","function":{"name":"Read","description":"Read a file","parameters":{"type":"object"}}}]
	}`)

	raw, _ := BuildKiroPayloadFromOpenAI(request, "claude-opus-5", "profile", "AI_EDITOR", modelcapabilities.Capability{}, "")
	var payload KiroPayload
	if err := json.Unmarshal(raw, &payload); err != nil {
		t.Fatalf("invalid payload: %v", err)
	}
	if got := len(payload.ConversationState.History[1].AssistantResponseMessage.ToolUses); got != 1 {
		t.Fatalf("declared tool uses = %d, want 1", got)
	}
	if got := len(payload.ConversationState.CurrentMessage.UserInputMessage.UserInputMessageContext.ToolResults); got != 1 {
		t.Fatalf("declared tool results = %d, want 1", got)
	}
}

func TestBuildKiroPayloadFlattensOnlyUndeclaredToolPairs(t *testing.T) {
	request := []byte(`{
		"messages":[
			{"role":"user","content":"go"},
			{"role":"assistant","content":null,"tool_calls":[{"id":"a","type":"function","function":{"name":"Read","arguments":"{\"p\":\"x\"}"}}]},
			{"role":"tool","tool_call_id":"a","content":"x-body"},
			{"role":"assistant","content":null,"tool_calls":[{"id":"b","type":"function","function":{"name":"mcp__gone__tool","arguments":"{}"}}]},
			{"role":"tool","tool_call_id":"b","content":"gone-body"},
			{"role":"assistant","content":"both","tool_calls":[
				{"id":"c","type":"function","function":{"name":"Read","arguments":"{\"p\":\"y\"}"}},
				{"id":"d","type":"function","function":{"name":"mcp__gone__tool","arguments":"{}"}}
			]},
			{"role":"tool","tool_call_id":"c","content":"y-body"},
			{"role":"tool","tool_call_id":"d","content":"d-body"}
		],
		"tools":[{"type":"function","function":{"name":"Read","description":"Read","parameters":{"type":"object"}}}]
	}`)
	raw, _ := BuildKiroPayloadFromOpenAI(request, "claude-opus-5", "profile", "AI_EDITOR", modelcapabilities.Capability{}, "")
	var payload KiroPayload
	if err := json.Unmarshal(raw, &payload); err != nil {
		t.Fatalf("invalid payload: %v", err)
	}
	h := payload.ConversationState.History
	current := payload.ConversationState.CurrentMessage.UserInputMessage
	toolUseIDs := func(m *KiroAssistantResponseMessage) string {
		ids := make([]string, 0, len(m.ToolUses))
		for _, toolUse := range m.ToolUses {
			ids = append(ids, toolUse.ToolUseID)
		}
		return strings.Join(ids, ",")
	}
	toolResultIDs := func(m *KiroUserInputMessage) string {
		if m.UserInputMessageContext == nil {
			return ""
		}
		ids := make([]string, 0, len(m.UserInputMessageContext.ToolResults))
		for _, result := range m.UserInputMessageContext.ToolResults {
			ids = append(ids, result.ToolUseID)
		}
		return strings.Join(ids, ",")
	}

	if got := toolUseIDs(h[1].AssistantResponseMessage); got != "a" {
		t.Errorf("history[1] tool uses = %q, want a", got)
	}
	if got := toolResultIDs(h[2].UserInputMessage); got != "a" {
		t.Errorf("history[2] tool results = %q, want a", got)
	}
	if got := toolUseIDs(h[3].AssistantResponseMessage); got != "" || !strings.Contains(h[3].AssistantResponseMessage.Content, `Historical unavailable tool call (not an executable call): {"toolUseId":"b","name":"mcp__gone__tool"`) {
		t.Errorf("history[3] tool uses = %q content = %q, want b flattened", got, h[3].AssistantResponseMessage.Content)
	}
	if got := toolResultIDs(h[4].UserInputMessage); got != "" || !strings.Contains(h[4].UserInputMessage.Content, `Historical unavailable tool result: {"content":[{"text":"gone-body"}],"status":"success","toolUseId":"b"}`) {
		t.Errorf("history[4] tool results = %q content = %q, want b flattened", got, h[4].UserInputMessage.Content)
	}
	if got := toolUseIDs(h[5].AssistantResponseMessage); got != "c" || !strings.Contains(h[5].AssistantResponseMessage.Content, `"toolUseId":"d"`) {
		t.Errorf("history[5] tool uses = %q content = %q, want c kept and d flattened", got, h[5].AssistantResponseMessage.Content)
	}
	if got := toolResultIDs(&current); got != "c" || !strings.Contains(current.Content, `"toolUseId":"d"`) || strings.Contains(current.Content, `"toolUseId":"c"`) {
		t.Errorf("current tool results = %q content = %q, want c kept and d flattened", got, current.Content)
	}
}
