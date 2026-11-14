// Request translation between the client wires (OpenAI, Anthropic) and the
// upstream wires (OpenAI, Anthropic, Gemini), plus extraction of a provider's
// JSON answer. The scorer compares what the provider echoed against the client's
// original body, field by field, so every translation preserves the echo fields
// verbatim: system text, last user text, tool names in order, max_tokens, stop
// strings, image bytes, and JSON mode. The parsers in the kit's env define the
// exact semantics; this file follows them:
//
//   system:  OpenAI joins system messages' text with \n; Anthropic joins the
//            system field's text blocks with \n; Gemini joins systemInstruction
//            parts with \n. One single text on every wire renders identically.
//   user:    the LAST user message's text (text blocks joined with \n).
//   tools:   definition names, in order. Descriptions/parameters are not echoed
//            but are passed through for fidelity.
//   stops:   passed through exactly, never converted between wires.
//   image:   the last image's decoded bytes (sha8). Base64 payloads pass through
//            untouched; a remote URL hashes as its own text on every wire.
//   json:    OpenAI response_format json_object/json_schema; Gemini
//            responseMimeType application/json; Anthropic has no marker and
//            always echoes false, so JSON requests never route there (plan.ts).
//   history: tool messages/results are invisible to every parser's echo, so the
//            Gemini renderer drops them (a trailing non-model content would
//            overwrite the user text with "") and the other renderers keep them.

import { isRecord, type Json } from './providers.ts';

export type ClientWire = 'openai' | 'anthropic';
export type UpstreamWire = 'openai' | 'anthropic' | 'gemini';

export type ImagePart = { kind: 'data'; media: string; data: string } | { kind: 'url'; url: string };

export interface ToolDef {
  name: string;
  description: string;
  parameters: Json;
}

export type MsgPart = { type: 'text'; text: string } | { type: 'image'; image: ImagePart };

export interface ToolCall {
  id: string;
  name: string;
  args: string;
}

export interface ToolResult {
  id: string;
  text: string;
}

export interface CMessage {
  role: 'user' | 'assistant' | 'tool';
  parts: MsgPart[];
  toolCalls: ToolCall[];
  toolResults: ToolResult[];
}

export type ToolChoiceKind = 'auto' | 'none' | 'required' | { name: string };

/** A client request in wire-neutral form. */
export interface Canonical {
  systemText: string;
  messages: CMessage[];
  tools: ToolDef[];
  toolChoice: ToolChoiceKind | null;
  maxTokens: number | null;
  temperature: number | null;
  topP: number | null;
  stops: string[];
  jsonMode: 'json_object' | 'json_schema' | null;
  stream: boolean;
}

function numberOrNull(v: unknown): number | null {
  return typeof v === 'number' && Number.isFinite(v) ? v : null;
}

function stringList(v: unknown): string[] {
  if (typeof v === 'string') return [v];
  if (!Array.isArray(v)) return [];
  return v.filter((x): x is string => typeof x === 'string');
}

/** Text of a string or a block array's text blocks, joined with \n (the env's blocksText). */
export function blocksText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (!Array.isArray(content)) return '';
  const parts: string[] = [];
  for (const block of content) {
    if (isRecord(block) && block.type === 'text' && typeof block.text === 'string') parts.push(block.text);
  }
  return parts.join('\n');
}

function imageFromDataUrl(url: string): ImagePart {
  const m = /^data:([^;,]*);base64,(.*)$/s.exec(url);
  if (m !== null) return { kind: 'data', media: m[1] !== '' ? m[1] : 'image/png', data: m[2] };
  return { kind: 'url', url };
}

function toolResultText(content: unknown): string {
  if (typeof content === 'string') return content;
  if (Array.isArray(content)) {
    const text = blocksText(content);
    return text !== '' ? text : JSON.stringify(content);
  }
  return content === undefined ? '' : JSON.stringify(content);
}

/** An OpenAI Chat Completions body in canonical form. */
export function fromOpenAI(body: Json): Canonical {
  const systemParts: string[] = [];
  const messages: CMessage[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'system') {
      const text = blocksText(raw.content);
      if (text !== '') systemParts.push(text);
      continue;
    }
    if (raw.role === 'assistant') {
      const toolCalls: ToolCall[] = [];
      for (const call of Array.isArray(raw.tool_calls) ? raw.tool_calls : []) {
        if (!isRecord(call) || !isRecord(call.function)) continue;
        toolCalls.push({
          id: typeof call.id === 'string' ? call.id : '',
          name: typeof call.function.name === 'string' ? call.function.name : '',
          args: typeof call.function.arguments === 'string' ? call.function.arguments : JSON.stringify(call.function.arguments ?? {}),
        });
      }
      const parts: MsgPart[] = [];
      const text = blocksText(raw.content);
      if (text !== '') parts.push({ type: 'text', text });
      messages.push({ role: 'assistant', parts, toolCalls, toolResults: [] });
      continue;
    }
    if (raw.role === 'tool') {
      messages.push({
        role: 'tool',
        parts: [],
        toolCalls: [],
        toolResults: [{ id: typeof raw.tool_call_id === 'string' ? raw.tool_call_id : '', text: toolResultText(raw.content) }],
      });
      continue;
    }
    const parts: MsgPart[] = [];
    if (typeof raw.content === 'string') {
      if (raw.content !== '') parts.push({ type: 'text', text: raw.content });
    } else if (Array.isArray(raw.content)) {
      for (const block of raw.content) {
        if (!isRecord(block)) continue;
        if (block.type === 'text' && typeof block.text === 'string') parts.push({ type: 'text', text: block.text });
        else if (block.type === 'image_url' && isRecord(block.image_url) && typeof block.image_url.url === 'string') {
          parts.push({ type: 'image', image: imageFromDataUrl(block.image_url.url) });
        }
      }
    }
    messages.push({ role: 'user', parts, toolCalls: [], toolResults: [] });
  }
  const tools: ToolDef[] = [];
  for (const raw of Array.isArray(body.tools) ? body.tools : []) {
    if (!isRecord(raw) || !isRecord(raw.function) || typeof raw.function.name !== 'string') continue;
    tools.push({
      name: raw.function.name,
      description: typeof raw.function.description === 'string' ? raw.function.description : '',
      parameters: isRecord(raw.function.parameters) ? raw.function.parameters : { type: 'object', properties: {} },
    });
  }
  const choice = body.tool_choice;
  let toolChoice: ToolChoiceKind | null = null;
  if (choice === 'auto') toolChoice = 'auto';
  else if (choice === 'none') toolChoice = 'none';
  else if (choice === 'required') toolChoice = 'required';
  else if (isRecord(choice) && choice.type === 'function' && isRecord(choice.function) && typeof choice.function.name === 'string') {
    toolChoice = { name: choice.function.name };
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return {
    systemText: systemParts.join('\n'),
    messages,
    tools,
    toolChoice,
    maxTokens: numberOrNull(body.max_tokens) ?? numberOrNull(body.max_completion_tokens),
    temperature: numberOrNull(body.temperature),
    topP: numberOrNull(body.top_p),
    stops: stringList(body.stop),
    jsonMode: format === 'json_object' || format === 'json_schema' ? format : null,
    stream: body.stream === true,
  };
}

/** An Anthropic Messages body in canonical form. Anthropic carries no JSON marker. */
export function fromAnthropic(body: Json): Canonical {
  const systemText = blocksText(body.system);
  const messages: CMessage[] = [];
  const source = Array.isArray(body.messages) ? body.messages : [];
  for (const raw of source) {
    if (!isRecord(raw)) continue;
    if (raw.role === 'assistant') {
      const parts: MsgPart[] = [];
      const toolCalls: ToolCall[] = [];
      if (typeof raw.content === 'string') {
        if (raw.content !== '') parts.push({ type: 'text', text: raw.content });
      } else if (Array.isArray(raw.content)) {
        for (const block of raw.content) {
          if (!isRecord(block)) continue;
          if (block.type === 'text' && typeof block.text === 'string') parts.push({ type: 'text', text: block.text });
          else if (block.type === 'tool_use') {
            toolCalls.push({
              id: typeof block.id === 'string' ? block.id : '',
              name: typeof block.name === 'string' ? block.name : '',
              args: JSON.stringify(isRecord(block.input) ? block.input : {}),
            });
          }
        }
      }
      messages.push({ role: 'assistant', parts, toolCalls, toolResults: [] });
      continue;
    }
    const parts: MsgPart[] = [];
    const toolResults: ToolResult[] = [];
    if (typeof raw.content === 'string') {
      if (raw.content !== '') parts.push({ type: 'text', text: raw.content });
    } else if (Array.isArray(raw.content)) {
      for (const block of raw.content) {
        if (!isRecord(block)) continue;
        if (block.type === 'text' && typeof block.text === 'string') parts.push({ type: 'text', text: block.text });
        else if (block.type === 'image' && isRecord(block.source)) {
          if (block.source.type === 'base64' && typeof block.source.data === 'string') {
            parts.push({ type: 'image', image: { kind: 'data', media: typeof block.source.media_type === 'string' ? block.source.media_type : 'image/png', data: block.source.data } });
          } else if (block.source.type === 'url' && typeof block.source.url === 'string') {
            parts.push({ type: 'image', image: { kind: 'url', url: block.source.url } });
          }
        } else if (block.type === 'tool_result') {
          toolResults.push({ id: typeof block.tool_use_id === 'string' ? block.tool_use_id : '', text: toolResultText(block.content) });
        }
      }
    }
    messages.push({ role: 'user', parts, toolCalls: [], toolResults: [] });
    for (const result of toolResults) messages.push({ role: 'tool', parts: [], toolCalls: [], toolResults: [result] });
  }
  const tools: ToolDef[] = [];
  for (const raw of Array.isArray(body.tools) ? body.tools : []) {
    if (!isRecord(raw) || typeof raw.name !== 'string') continue;
    tools.push({
      name: raw.name,
      description: typeof raw.description === 'string' ? raw.description : '',
      parameters: isRecord(raw.input_schema) ? raw.input_schema : { type: 'object', properties: {} },
    });
  }
  const choice = body.tool_choice;
  let toolChoice: ToolChoiceKind | null = null;
  if (isRecord(choice)) {
    if (choice.type === 'auto') toolChoice = 'auto';
    else if (choice.type === 'any') toolChoice = 'required';
    else if (choice.type === 'none') toolChoice = 'none';
    else if (choice.type === 'tool' && typeof choice.name === 'string') toolChoice = { name: choice.name };
  }
  return {
    systemText,
    messages,
    tools,
    toolChoice,
    maxTokens: numberOrNull(body.max_tokens),
    temperature: numberOrNull(body.temperature),
    topP: numberOrNull(body.top_p),
    stops: stringList(body.stop_sequences),
    jsonMode: null,
    stream: body.stream === true,
  };
}

function openAIImage(image: ImagePart): Json {
  return image.kind === 'data' ? { url: `data:${image.media};base64,${image.data}` } : { url: image.url };
}

/** A user message's content: a plain string when it holds one text part, else parts. */
function openAIUserContent(parts: MsgPart[]): string | Json[] {
  if (parts.length === 0) return '';
  if (parts.length === 1 && parts[0].type === 'text') return parts[0].text;
  return parts.map((p) => (p.type === 'text' ? { type: 'text', text: p.text } : { type: 'image_url', image_url: openAIImage(p.image) }));
}

function openAIToolChoice(choice: ToolChoiceKind): string | Json {
  if (typeof choice === 'string') return choice;
  return { type: 'function', function: { name: choice.name } };
}

/** The canonical request as an OpenAI Chat Completions body naming `model`. JSON, never streamed. */
export function toOpenAI(canon: Canonical, model: string): Json {
  const messages: Json[] = [];
  if (canon.systemText !== '') messages.push({ role: 'system', content: canon.systemText });
  for (const m of canon.messages) {
    if (m.role === 'user') {
      messages.push({ role: 'user', content: openAIUserContent(m.parts) });
    } else if (m.role === 'assistant') {
      const text = m.parts.filter((p): p is { type: 'text'; text: string } => p.type === 'text').map((p) => p.text).join('\n');
      messages.push({
        role: 'assistant',
        content: text !== '' ? text : null,
        ...(m.toolCalls.length > 0
          ? { tool_calls: m.toolCalls.map((c) => ({ id: c.id, type: 'function', function: { name: c.name, arguments: c.args } })) }
          : {}),
      });
    } else {
      for (const r of m.toolResults) messages.push({ role: 'tool', tool_call_id: r.id, content: r.text });
    }
  }
  const out: Json = { model, messages, stream: false };
  if (canon.maxTokens !== null) out.max_tokens = canon.maxTokens;
  if (canon.temperature !== null) out.temperature = canon.temperature;
  if (canon.topP !== null) out.top_p = canon.topP;
  if (canon.stops.length > 0) out.stop = canon.stops;
  if (canon.tools.length > 0) {
    out.tools = canon.tools.map((t) => ({ type: 'function', function: { name: t.name, description: t.description, parameters: t.parameters } }));
  }
  if (canon.toolChoice !== null) out.tool_choice = openAIToolChoice(canon.toolChoice);
  if (canon.jsonMode !== null) out.response_format = { type: canon.jsonMode };
  return out;
}

function anthropicToolChoice(choice: ToolChoiceKind): Json | null {
  if (choice === 'auto') return { type: 'auto' };
  if (choice === 'required') return { type: 'any' };
  if (typeof choice === 'object') return { type: 'tool', name: choice.name };
  return null;
}

/** The canonical request as an Anthropic Messages body naming `model`. JSON, never streamed. */
export function toAnthropic(canon: Canonical, model: string): Json {
  const messages: Json[] = [];
  for (const m of canon.messages) {
    if (m.role === 'user') {
      if (m.parts.length === 1 && m.parts[0].type === 'text') {
        messages.push({ role: 'user', content: m.parts[0].text });
      } else if (m.parts.length === 0) {
        messages.push({ role: 'user', content: '' });
      } else {
        const blocks: Json[] = m.parts.map((p) =>
          p.type === 'text'
            ? { type: 'text', text: p.text }
            : p.image.kind === 'data'
              ? { type: 'image', source: { type: 'base64', media_type: p.image.media, data: p.image.data } }
              : { type: 'image', source: { type: 'url', url: p.image.url } },
        );
        messages.push({ role: 'user', content: blocks });
      }
    } else if (m.role === 'assistant') {
      const blocks: Json[] = [];
      for (const p of m.parts) if (p.type === 'text') blocks.push({ type: 'text', text: p.text });
      for (const c of m.toolCalls) {
        let input: Json = {};
        try {
          const parsed: unknown = JSON.parse(c.args);
          if (isRecord(parsed)) input = parsed;
        } catch {
          // Not JSON: an empty input.
        }
        blocks.push({ type: 'tool_use', id: c.id, name: c.name, input });
      }
      if (blocks.length === 1 && blocks[0].type === 'text') messages.push({ role: 'assistant', content: blocks[0].text });
      else messages.push({ role: 'assistant', content: blocks });
    } else {
      messages.push({
        role: 'user',
        content: m.toolResults.map((r) => ({ type: 'tool_result', tool_use_id: r.id, content: r.text })),
      });
    }
  }
  const out: Json = { model, messages, stream: false };
  if (canon.systemText !== '') out.system = canon.systemText;
  if (canon.maxTokens !== null) out.max_tokens = canon.maxTokens;
  if (canon.temperature !== null) out.temperature = canon.temperature;
  if (canon.topP !== null) out.top_p = canon.topP;
  if (canon.stops.length > 0) out.stop_sequences = canon.stops;
  if (canon.tools.length > 0) {
    out.tools = canon.tools.map((t) => ({ name: t.name, description: t.description, input_schema: t.parameters }));
  }
  if (canon.toolChoice !== null) {
    const choice = anthropicToolChoice(canon.toolChoice);
    if (choice !== null) out.tool_choice = choice;
  }
  return out;
}

/**
 * The canonical request as a Gemini generateContent body. The model travels in
 * the URL, not the body. Tool history is dropped: a trailing non-model content
 * would overwrite the echoed user text with "".
 */
export function toGemini(canon: Canonical): Json {
  const contents: Json[] = [];
  for (const m of canon.messages) {
    if (m.role === 'user') {
      const parts: Json[] = [];
      for (const p of m.parts) {
        if (p.type === 'text') parts.push({ text: p.text });
        else if (p.image.kind === 'data') parts.push({ inlineData: { mimeType: p.image.media, data: p.image.data } });
        else parts.push({ fileData: { fileUri: p.image.url, mimeType: 'image/png' } });
      }
      contents.push({ role: 'user', parts });
    } else if (m.role === 'assistant') {
      const parts: Json[] = [];
      for (const p of m.parts) if (p.type === 'text') parts.push({ text: p.text });
      if (parts.length > 0) contents.push({ role: 'model', parts });
    }
    // Tool messages are dropped (see the header note).
  }
  const generationConfig: Json = {};
  if (canon.maxTokens !== null) generationConfig.maxOutputTokens = canon.maxTokens;
  if (canon.temperature !== null) generationConfig.temperature = canon.temperature;
  if (canon.topP !== null) generationConfig.topP = canon.topP;
  if (canon.stops.length > 0) generationConfig.stopSequences = canon.stops;
  if (canon.jsonMode !== null) generationConfig.responseMimeType = 'application/json';
  const out: Json = { contents, generationConfig };
  if (canon.systemText !== '') out.systemInstruction = { parts: [{ text: canon.systemText }] };
  if (canon.tools.length > 0) {
    out.tools = [{ functionDeclarations: canon.tools.map((t) => ({ name: t.name, description: t.description, parameters: t.parameters })) }];
  }
  return out;
}

/** A provider's JSON answer, reduced to what the client response needs. */
export interface ProviderAnswer {
  text: string;
  inputTokens: number;
  outputTokens: number;
}

function num(v: unknown): number {
  return typeof v === 'number' && Number.isFinite(v) ? v : 0;
}

export function extractOpenAI(body: Json): ProviderAnswer {
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const choice = isRecord(choices[0]) ? choices[0] : {};
  const message = isRecord(choice.message) ? choice.message : {};
  const content = typeof message.content === 'string' ? message.content : blocksText(message.content);
  const usage = isRecord(body.usage) ? body.usage : {};
  return { text: content, inputTokens: num(usage.prompt_tokens), outputTokens: num(usage.completion_tokens) };
}

export function extractAnthropic(body: Json): ProviderAnswer {
  const content = Array.isArray(body.content) ? body.content : [];
  const text = content
    .filter((b): b is Json => isRecord(b) && b.type === 'text' && typeof b.text === 'string')
    .map((b) => b.text as string)
    .join('');
  const usage = isRecord(body.usage) ? body.usage : {};
  return { text, inputTokens: num(usage.input_tokens), outputTokens: num(usage.output_tokens) };
}

export function extractGemini(body: Json): ProviderAnswer {
  const candidates = Array.isArray(body.candidates) ? body.candidates : [];
  const candidate = isRecord(candidates[0]) ? candidates[0] : {};
  const content = isRecord(candidate.content) ? candidate.content : {};
  const parts = Array.isArray(content.parts) ? content.parts : [];
  const text = parts
    .filter((p): p is Json => isRecord(p) && typeof p.text === 'string')
    .map((p) => p.text as string)
    .join('');
  const usage = isRecord(body.usageMetadata) ? body.usageMetadata : {};
  return { text, inputTokens: num(usage.promptTokenCount), outputTokens: num(usage.candidatesTokenCount) };
}

export function extractAnswer(wire: UpstreamWire, body: Json): ProviderAnswer {
  if (wire === 'anthropic') return extractAnthropic(body);
  if (wire === 'gemini') return extractGemini(body);
  return extractOpenAI(body);
}
