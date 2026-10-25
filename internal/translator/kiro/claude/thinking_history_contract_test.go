package claude

import (
	"encoding/json"
	"reflect"
	"testing"

	"github.com/nguyenha935/kiro-cpa-plugin/internal/modelcapabilities"
	"github.com/tidwall/gjson"
)

func TestThinkingHistoryPreservesEmptySignedWhitespaceAndRedactedBlocks(t *testing.T) {
	t.Parallel()
	for _, test := range []struct {
		name, block, native string
	}{
		{"signed-whitespace", `{"type":"thinking","thinking":"  reasoning\n\n","signature":"signature-one"}`, `{"reasoningText":{"text":"  reasoning\n\n","signature":"signature-one"}}`},
		{"signed-empty", `{"type":"thinking","thinking":"","signature":"signature-one"}`, `{"reasoningText":{"text":"","signature":"signature-one"}}`},
		{"redacted", `{"type":"redacted_thinking","data":"opaque-redacted"}`, `{"redactedContent":"opaque-redacted"}`},
	} {
		t.Run(test.name, func(t *testing.T) {
			body := []byte(`{"messages":[{"role":"user","content":"Think"},{"role":"assistant","content":[` + test.block + `,{"type":"text","text":"Answer"}]},{"role":"user","content":"Continue"}]}`)
			payload, _ := BuildKiroPayload(body, "claude-opus-5.5", "profile", "AI_EDITOR", modelcapabilities.Capability{}, "xhigh")
			assistant := gjson.GetBytes(payload, "conversationState.history.1.assistantResponseMessage")
			var got, want any
			if err := json.Unmarshal([]byte(assistant.Get("reasoningContent").Raw), &got); err != nil {
				t.Fatalf("decode reasoning: %v; assistant=%s", err, assistant.Raw)
			}
			if err := json.Unmarshal([]byte(test.native), &want); err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(got, want) {
				t.Fatalf("reasoning = %#v, want %#v", got, want)
			}
			if assistant.Get("content").String() != "Answer" {
				t.Fatalf("visible answer changed: %s", assistant.Raw)
			}
		})
	}
}

func TestThinkingHistoryKeepsContentAndToolsWithTheirSignedBlock(t *testing.T) {
	t.Parallel()
	body := []byte(`{"messages":[
		{"role":"user","content":"Inspect both files"},
		{"role":"assistant","content":[
			{"type":"thinking","thinking":"first\n","signature":"signature-one"},
			{"type":"text","text":"Read first"},
			{"type":"tool_use","id":"call_1","name":"Read","input":{"path":"first.txt"}},
			{"type":"thinking","thinking":"","signature":"signature-two"},
			{"type":"text","text":"Read second"},
			{"type":"tool_use","id":"call_2","name":"Read","input":{"path":"second.txt"}}
		]},
		{"role":"user","content":[
			{"type":"tool_result","tool_use_id":"call_1","content":"first result"},
			{"type":"tool_result","tool_use_id":"call_2","content":"second result"}
		]}
	],"tools":[{"name":"Read","description":"Read a file","input_schema":{"type":"object","properties":{"path":{"type":"string"}}}}]}`)
	raw, _ := BuildKiroPayload(body, "claude-opus-5.5", "profile", "AI_EDITOR", modelcapabilities.Capability{}, "xhigh")
	history := gjson.GetBytes(raw, "conversationState.history").Array()
	if len(history) != 3 {
		t.Fatalf("history messages = %d, want user and two assistant entries: %s", len(history), raw)
	}
	for i, want := range []string{
		`{"content":"Read first","reasoningContent":{"reasoningText":{"text":"first\n","signature":"signature-one"}},"toolUses":[{"toolUseId":"call_1","name":"Read","input":{"path":"first.txt"}}]}`,
		`{"content":"Read second","reasoningContent":{"reasoningText":{"text":"","signature":"signature-two"}},"toolUses":[{"toolUseId":"call_2","name":"Read","input":{"path":"second.txt"}}]}`,
	} {
		got := history[i+1].Get("assistantResponseMessage")
		if !reflect.DeepEqual(got.Value(), gjson.Parse(want).Value()) {
			t.Fatalf("assistant %d = %s, want %s", i, got.Raw, want)
		}
	}
	results := gjson.GetBytes(raw, "conversationState.currentMessage.userInputMessage.userInputMessageContext.toolResults").Array()
	if len(results) != 2 || results[0].Get("toolUseId").String() != "call_1" || results[1].Get("toolUseId").String() != "call_2" {
		t.Fatalf("tool result pairing changed: %s", raw)
	}
}
