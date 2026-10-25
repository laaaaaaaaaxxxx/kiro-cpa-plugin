package executor

import (
	"io"
	"net/http"
	"strings"
	"testing"

	cliproxyexecutor "github.com/router-for-me/CLIProxyAPI/v7/sdk/cliproxy/executor"
	sdktranslator "github.com/router-for-me/CLIProxyAPI/v7/sdk/translator"
	"github.com/tidwall/gjson"
)

func TestThinkingSignatureRejectionDoesNotRetryOrDropHistory(t *testing.T) {
	for _, stream := range []bool{false, true} {
		name := "buffered"
		if stream {
			name = "stream"
		}
		t.Run(name, func(t *testing.T) {
			requests := 0
			var sent []byte
			stubUpstream(t, func(request *http.Request) (*http.Response, error) {
				requests++
				var err error
				sent, err = io.ReadAll(request.Body)
				if err != nil {
					return nil, err
				}
				return upstreamStatus(http.StatusBadRequest, nil, `{"message":"THINKING_SIGNATURE_INVALID","reason":"THINKING_SIGNATURE_INVALID"}`), nil
			})
			body := []byte(`{"messages":[{"role":"user","content":"Think"},{"role":"assistant","content":[{"type":"thinking","thinking":"reasoning\n\n","signature":"invalid-upstream-signature"},{"type":"text","text":"Answer"}]},{"role":"user","content":"Continue"}]}`)
			original := string(body)
			request := cliproxyexecutor.Request{Model: "claude-haiku-4.5", Payload: body, Format: sdktranslator.FormatClaude}
			options := cliproxyexecutor.Options{SourceFormat: sdktranslator.FormatClaude, OriginalRequest: body}
			auth := builderIDAuth("thinking-invalid-" + name)
			executor := NewKiroExecutor(nil)
			var err error
			if stream {
				_, err = executor.ExecuteStream(t.Context(), auth, request, options)
			} else {
				_, err = executor.Execute(t.Context(), auth, request, options)
			}
			if statusOf(err) != http.StatusBadRequest || !strings.Contains(err.Error(), "THINKING_SIGNATURE_INVALID") {
				t.Fatalf("error = %v, want original signature rejection", err)
			}
			if requests != 1 {
				t.Fatalf("upstream requests = %d, want 1 without stripping history", requests)
			}
			reasoning := gjson.GetBytes(sent, "conversationState.history.1.assistantResponseMessage.reasoningContent.reasoningText")
			if reasoning.Get("text").String() != "reasoning\n\n" || reasoning.Get("signature").String() != "invalid-upstream-signature" {
				t.Fatalf("outbound reasoning changed: %s", reasoning.Raw)
			}
			if string(body) != original {
				t.Fatal("caller history was mutated")
			}
		})
	}
}
