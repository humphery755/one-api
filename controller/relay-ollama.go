package controller

import (
	"bufio"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"one-api/common"
	"strings"
	"time"

	"github.com/gin-gonic/gin"
)

// https://cloud.Ollama.com/document/product/1729/97732

type OllamaChatRequest struct {
	Model    string `json:"model"`
	Prompt   string `json:"prompt"`
	System   string `json:"system"`
	Template string `json:"template"`
	Context  []int  `json:"context,omitempty"`
	Stream   *bool  `json:"stream,omitempty"`

	Options map[string]interface{} `json:"options"`
}

type OllamaChatResponse struct {
	Model     string    `json:"model"`
	CreatedAt time.Time `json:"created_at"`
	Response  string    `json:"response"`

	Done    bool  `json:"done"`
	Context []int `json:"context,omitempty"`

	TotalDuration      time.Duration `json:"total_duration,omitempty"`
	LoadDuration       time.Duration `json:"load_duration,omitempty"`
	PromptEvalCount    int           `json:"prompt_eval_count,omitempty"`
	PromptEvalDuration time.Duration `json:"prompt_eval_duration,omitempty"`
	EvalCount          int           `json:"eval_count,omitempty"`
	EvalDuration       time.Duration `json:"eval_duration,omitempty"`
}

func requestOpenAI2Ollama(request GeneralOpenAIRequest) *OllamaChatRequest {
	req := OllamaChatRequest{
		Model:  request.Model,
		Prompt: fmt.Sprintf("%s", request.Prompt),
		Stream: &request.Stream,
	}
	for i := 0; i < len(request.Messages); i++ {
		message := request.Messages[i]
		if message.Role == "system" {
			req.System = message.Content
		} else if message.Role == "user" {
			req.Prompt = message.Content
		}
	}

	return &req
}

func responseOllama2OpenAI(response *OllamaChatResponse) *OpenAITextResponse {
	fullTextResponse := OpenAITextResponse{
		Object:  "chat.completion",
		Created: common.GetTimestamp(),
	}
	var FinishReason = ""
	if response.Done == true {
		FinishReason = "stop"
	}

	choice := OpenAITextResponseChoice{
		Index: 0,
		Message: Message{
			Role:    "assistant",
			Content: response.Response,
		},
		FinishReason: FinishReason,
	}
	fullTextResponse.Choices = append(fullTextResponse.Choices, choice)
	return &fullTextResponse
}

func streamResponseOllama2OpenAI(OllamaResponse *OllamaChatResponse) *ChatCompletionsStreamResponse {
	var choice ChatCompletionsStreamResponseChoice
	if OllamaResponse.Done {
		choice.FinishReason = &stopFinishReason
	}
	choice.Delta.Content = OllamaResponse.Response
	response := ChatCompletionsStreamResponse{
		Object:  "chat.completion.chunk",
		Created: common.GetTimestamp(),
		Model:   OllamaResponse.Model,
		Choices: []ChatCompletionsStreamResponseChoice{choice},
	}
	return &response
}

func ollamaStreamHandler(c *gin.Context, resp *http.Response) (*OpenAIErrorWithStatusCode, string) {

	var responseText string
	scanner := bufio.NewScanner(resp.Body)

	scanner.Split(func(data []byte, atEOF bool) (advance int, token []byte, err error) {
		if atEOF && len(data) == 0 {
			return 0, nil, nil
		}
		if i := strings.Index(string(data), "\n"); i >= 0 {
			return i + 1, data[0:i], nil
		}
		if atEOF {
			return len(data), data, nil
		}
		return 0, nil, nil
	})
	dataChan := make(chan string)
	stopChan := make(chan bool)
	go func() {
		for scanner.Scan() {
			data := scanner.Text()

			if len(data) < 5 { // ignore blank line or wrong format
				continue
			}
			// common.LogInfo(c, ":=>"+data)
			// data = data[5:]
			dataChan <- data
		}
		stopChan <- true
	}()
	setEventStreamHeaders(c)
	id := common.GetUUID()
	c.Stream(func(w io.Writer) bool {
		select {
		case data := <-dataChan:
			var OllamaResponse OllamaChatResponse
			err := json.Unmarshal([]byte(data), &OllamaResponse)
			if err != nil {
				common.SysError("error unmarshalling stream response: " + err.Error())
				return true
			}
			response := streamResponseOllama2OpenAI(&OllamaResponse)
			if len(response.Choices) != 0 {
				responseText += response.Choices[0].Delta.Content
			}
			response.Id = id
			jsonResponse, err := json.Marshal(response)
			if err != nil {
				common.SysError("error marshalling stream response: " + err.Error())
				return true
			}
			c.Render(-1, common.CustomEvent{Data: "data: " + string(jsonResponse)})
			return true
		case <-stopChan:
			c.Render(-1, common.CustomEvent{Data: "data: [DONE]"})
			return false
		}
	})
	err := resp.Body.Close()
	if err != nil {
		return errorWrapper(err, "close_response_body_failed", http.StatusInternalServerError), ""
	}
	return nil, responseText
}

func ollamaHandler(c *gin.Context, resp *http.Response) (*OpenAIErrorWithStatusCode, *Usage) {
	var ollamaResponse OllamaChatResponse

	responseBody, err := io.ReadAll(resp.Body)
	if err != nil {
		return errorWrapper(err, "read_response_body_failed", http.StatusInternalServerError), nil
	}
	err = resp.Body.Close()
	if err != nil {
		return errorWrapper(err, "close_response_body_failed", http.StatusInternalServerError), nil
	}
	err = json.Unmarshal(responseBody, &ollamaResponse)
	if err != nil {
		return errorWrapper(err, "unmarshal_response_body_failed", http.StatusInternalServerError), nil
	}
	/*if ollamaResponse.Error.Code != 0 {
		return &OpenAIErrorWithStatusCode{
			OpenAIError: OpenAIError{
				Message: ollamaResponse.Error.Message,
				Code:    ollamaResponse.Error.Code,
			},
			StatusCode: resp.StatusCode,
		}, nil
	}*/
	fullTextResponse := responseOllama2OpenAI(&ollamaResponse)
	jsonResponse, err := json.Marshal(fullTextResponse)
	if err != nil {
		return errorWrapper(err, "marshal_response_body_failed", http.StatusInternalServerError), nil
	}
	c.Writer.Header().Set("Content-Type", "application/json")
	c.Writer.WriteHeader(resp.StatusCode)
	_, err = c.Writer.Write(jsonResponse)
	return nil, &fullTextResponse.Usage
}
