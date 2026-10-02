export const meta = {
  name: 'ko-train-data',
  description: 'Generate ~3,400 Korean outage-report training rows (10 styles x 2 settings) and audit labels with two independent lenses',
  phases: [
    { title: 'Generate', detail: '20 chunks x 170 rows, written as 5 part files each' },
    { title: 'Audit', detail: 'label-rule and span/style auditors per chunk; fixes returned as data' },
  ],
}

const RAW = 'F:\\Develop\\NeedleTest\\ko\\out\\raw\\'
const STYLES = [
  '표준어로 차분하게 신고하는 주민',
  '경상도 사투리를 쓰는 어르신',
  '전라도 사투리를 쓰는 어르신',
  '충청도 사투리를 쓰는 주민',
  '급하고 당황해서 문장이 끊기는 주민',
  '맞춤법과 띄어쓰기가 틀린 문자 메시지',
  '음성 인식(STT) 오류가 섞인 문장',
  '현장 작업자의 짧은 업무용 보고 (약어, 설비 번호 포함)',
  '한국어가 서툰 외국인 주민',
  '영어로 신고하는 외국인 주민',
]
const SETTINGS = [
  { key: 'a', text: '농어촌·산간·해안 지역 (리·면·읍, 마을회관, 경로당, 논밭, 과수원, 축사, 비닐하우스, 저수지, 항구, 어시장, 펜션, 등산로 입구, 국도변)', disasters: '태풍, 폭우, 산사태, 강풍, 해일 주의보' },
  { key: 'b', text: '도시·공단 지역 (동, 아파트 단지와 동호수, 빌라, 오피스텔, 상가, 전통시장, 학교, 병원, 지하철역, 교차로, 지하차도, 공단, 물류창고, 대학교 기숙사)', disasters: '산불 연기, 강풍, 폭설, 낙뢰, 폭염, 태풍' },
]
const TOOL = '{"name": "report_incident", "description": "Record a power outage or electrical equipment damage report.", "parameters": {"type": "object", "properties": {"incident_type": {"type": "string", "enum": ["outage", "pole_down", "line_down", "transformer_noise", "spark", "fire"], "description": "outage, pole_down, line_down, transformer_noise, spark, or fire."}, "location": {"type": "string", "description": "Where it happened, copied as the reporter said it."}, "households": {"type": "integer", "description": "Number of affected households, only when stated.", "minimum": 1, "maximum": 100000}, "hazard": {"type": "string", "enum": ["electrocution", "fire"], "description": "Immediate danger, only when the reporter describes one."}}, "required": ["incident_type", "location"]}}'
const RULES = `- 태풍·폭우·산불 같은 재난 상황에서 정전이나 전력설비 피해를 신고하는 문장입니다.
- evidence는 신고 문장이 말하는 내용을 2~8단어 영어로 요약한 구절입니다. 한글을 쓰지 않습니다.
  incident_type의 근거를 담고, households나 hazard를 넣었다면 그 근거도 담습니다.
  예: "power is out", "pole fell, says it is dangerous", "power out, 30 households stated".
  answers가 []이면 무엇에 대한 문장인지 씁니다. 예: "asks about the bill", "says the power is fine".
- arguments의 키와 incident_type, hazard 값은 스키마의 영어 값만 씁니다.
- incident_type: outage(전기가 안 들어옴), pole_down(전봇대·전주가 쓰러지거나 기울어짐),
  line_down(전선이 끊어지거나 늘어지거나 떨어짐), transformer_noise(변압기 소음),
  spark(불꽃·스파크, 불은 나지 않음), fire(설비에 불이 남).
- 신고 하나에 호출 하나만 만듭니다. 여러 현상이 함께 나오면 가장 심각한 설비 피해 하나로 정합니다:
  fire > pole_down > line_down > spark > transformer_noise > outage.
  예: 전봇대가 쓰러지고 전선이 바닥에 있으면 pole_down, 전선에서 불꽃만 튀면 spark.
- location은 query 안에 글자 그대로 들어 있는 장소 표현입니다(번역하거나 다듬지 않음).
  앞·옆·뒤편·근처·주변 같은 위치 말은 포함하고, 끝의 조사(에, 에서, 의 등)와
  설비 이름(전봇대, 전주, 변압기, 전선 등)은 뺍니다(예: "공원 입구 전주에서 불꽃이" → "공원 입구").
  영어는 near, by, on 같은 전치사를 빼고 명사구만 씁니다(예: "the school").
- households는 가구·세대 수를 직접 말한 경우에만 넣습니다(정수). 아파트 동·호수 같은 숫자는 가구 수가 아닙니다.
- hazard는 감전이나 화재 위험을 직접 말한 경우에만 넣습니다. 불꽃이 튄다는 말만으로는 넣지 않고,
  불이 날 것 같다고 하면 fire, 위험하다거나 감전될 것 같다고 하면 electrocution입니다.
  실제로 불이 났으면 incident_type과 hazard 모두 fire입니다.
- 지명은 실제와 가상을 섞고 매번 다르게 합니다 (OO리, OO동, 아파트 동호수, 시장, 학교, 다리 등).
  같은 지명을 되풀이하지 않고, 흔한 이름(행복, 햇살, 햇빛, 한빛, 푸른, 새마을 등)은 피합니다.
  문장 길이와 구성도 두세 단어 문자부터 두세 문장 설명까지 다양하게 합니다.
- 거절 행(answers [])은 정전·설비 피해와 무관하거나(요금 문의, 명의 변경, 날씨 등), 정전이 아니라는 내용입니다.`
const FORBID = `- These are an evaluation set that must stay unseen: never open, read, search, or list ko/data/test_human.jsonl, ko/make_test_data.py, or anything under ko/results/. Do not read any other repository file either; everything you need is in this message.
- Do not run shell commands.`

const chunks = []
STYLES.forEach((style, i) => SETTINGS.forEach(s => chunks.push({ id: 's' + String(i + 1).padStart(2, '0') + s.key, style, setting: s })))
chunks.push({ id: 'pilot', pilot: true })

const filesOf = c => c.pilot ? [RAW + 'pilot.jsonl'] : [1, 2, 3, 4, 5].map(k => RAW + c.id + '_p' + k + '.jsonl')

const GEN_SCHEMA = {
  type: 'object',
  properties: {
    files: { type: 'array', items: { type: 'string' } },
    rows: { type: 'integer' },
    refusals: { type: 'integer' },
    notes: { type: 'string' },
  },
  required: ['files', 'rows', 'refusals', 'notes'],
}
const CALL = {
  type: 'object',
  properties: {
    name: { type: 'string', enum: ['report_incident'] },
    arguments: {
      type: 'object',
      properties: {
        incident_type: { type: 'string', enum: ['outage', 'pole_down', 'line_down', 'transformer_noise', 'spark', 'fire'] },
        location: { type: 'string' },
        households: { type: 'integer' },
        hazard: { type: 'string', enum: ['electrocution', 'fire'] },
      },
      required: ['incident_type', 'location'],
      additionalProperties: false,
    },
  },
  required: ['name', 'arguments'],
}
const FIX_SCHEMA = {
  type: 'object',
  properties: {
    checked: { type: 'integer', description: 'number of rows you read and checked' },
    missing_files: { type: 'array', items: { type: 'string' } },
    fixes: {
      type: 'array',
      items: {
        type: 'object',
        properties: {
          file: { type: 'string', description: 'file path exactly as given in the task' },
          line: { type: 'integer', description: '1-based line number in that file' },
          query: { type: 'string', description: 'the row query copied exactly' },
          action: { type: 'string', enum: ['fix', 'drop'] },
          answers: { type: 'array', items: CALL, description: 'for fix: the full corrected answers ([] for a refusal)' },
          evidence: { type: 'string', description: 'for fix: corrected English evidence phrase, 2-8 words, no Korean' },
          reason: { type: 'string' },
        },
        required: ['file', 'line', 'query', 'action', 'reason'],
      },
    },
  },
  required: ['checked', 'missing_files', 'fixes'],
}

function genPrompt(c) {
  const files = filesOf(c)
  return `You are generating synthetic training data for fine-tuning a small on-device model that turns disaster-time power outage / electrical equipment damage reports into a report_incident tool call.

## Hard constraints
${FORBID}
- Only use the Write tool, to create exactly these 5 files (34 rows each, one Write call per file):
${files.map(f => '  ' + f).join('\n')}

## This chunk
- 화자 스타일 (모든 행): ${c.style}
- 지역 배경: ${c.setting.text}
- 재난 상황: ${c.setting.disasters}
- 행 구성 (총 170행):
  - 거절 20행 (answers []): 요금·납부·고지서·명의 변경·이사·날씨·복구 확인·정전 아님 등 다양하게
  - outage 55행 (그중 15행쯤은 가구·세대 수를 직접 말함)
  - line_down 25행, pole_down 20행, spark 20행, transformer_noise 15행, fire 15행
  - 위험(감전·화재)을 직접 말하는 행 30행쯤 (hazard 있음), 설비 피해인데 위험을 말하지 않는 행도 30행 이상 (hazard 없음, 불꽃만 튀는 문장 포함)
  - 두 가지 이상 현상이 함께 나오는 행 15행쯤 (우선순위 규칙 적용)
  - 길이: 두세 단어 문자 30행쯤, 한 문장 90행쯤, 두세 문장 50행쯤
- 스타일의 말투가 모든 행에서 실제로 드러나야 합니다. 영어 스타일이면 query, location 모두 영어입니다.

## Row format
JSONL, one JSON object per line, UTF-8, no array, no comments:
{"query": "<신고 문장>", "evidence": "<영어 요약>", "answers": [{"name": "report_incident", "arguments": {...}}], "style": "${c.style}"}

Tool schema:
${TOOL}

## 규칙
${RULES}

## Before writing each file, check every row
- location appears character-for-character inside query, without particles or equipment nouns.
- Only the keys incident_type, location, households, hazard; enum values exactly as in the schema; households is a JSON integer.
- evidence has no Korean characters; refusal rows have "answers": [].
- No duplicate query and no repeated place name across all 170 rows.

Return the file paths, total rows, number of refusal rows, and notes on any rule that was hard to apply.`
}

const LENS = {
  label: `Lens: LABEL RULES. For every row check:
- incident_type matches the definitions, and when several phenomena appear the precedence fire > pole_down > line_down > spark > transformer_noise > outage was applied.
- hazard is present only when danger (electrocution or fire risk) is directly stated; sparks alone get no hazard; "불이 날 것 같다" → fire; "위험하다/감전될 것 같다" → electrocution; an actual fire → incident_type fire and hazard fire.
- households is present only when a count of households/세대/가구/수용가 is directly stated, and its value matches the stated number; apartment 동/호수, 번지, 단지 numbers are not household counts.
- refusals: answers [] exactly when no outage or equipment damage is being reported (unrelated questions, "not an outage", "power came back"); a row that does report an incident must have a call.
- evidence is English, 2-8 words, and does not contradict the label (it should state the basis for the type, and for households/hazard when present).`,
  span: `Lens: LOCATION SPAN, STYLE, AND DUPLICATES. For every row check:
- location is a character-for-character substring of query and is the place phrase: it keeps position words (앞, 옆, 뒤편, 근처, 주변, 인근, 쪽, 입구 ...) but not trailing particles (에, 에서, 의, 이요, 예요 ...) and not equipment nouns (전봇대, 전주, 변압기, 전선, TR ...); English drops prepositions (near, by, on, at, in) and keeps the noun phrase.
- location is the most specific place the reporter gave (not a broader region when a more specific place is stated in the same report).
- the row really sounds like the stated speaker style; natural Korean for that style (dialect endings, typos, STT misrecognitions, field shorthand, broken Korean, or English as appropriate).
- duplicates: drop a row whose query is the same as or a trivial variation of an earlier row in these files, or that reuses an earlier row's place name.`,
}

function auditPrompt(c, lens) {
  const files = filesOf(c)
  return `You are auditing synthetic training rows for a Korean outage-report tool-calling model. Rows are JSONL: {"query", "evidence", "answers", "style"}.

## Hard constraints
${FORBID}
- Use only the Read tool, on exactly these files (read every line; if a file is missing, list it in missing_files):
${files.map(f => '  ' + f).join('\n')}

## The labeling rules the rows must follow
Tool schema:
${TOOL}

${RULES}

## Your task
${LENS[lens]}

Report only clear violations of the rules; when a label is defensible, leave it alone. For each violation return one fix:
- action "fix" with the full corrected answers (use [] to turn a row into a refusal) and a corrected evidence phrase, when the row is usable after correcting its label;
- action "drop" when the query itself is the problem (wrong style, unnatural, duplicate, ambiguous so that no label is clearly right).
Never change the query text. Give the file path exactly as listed above, the 1-based line number as shown by the Read tool, and the query copied exactly. Set checked to the number of rows you read.`
}

const results = await pipeline(
  chunks,
  async (c) => {
    if (c.pilot) return { files: filesOf(c), rows: 100, refusals: 15, notes: 'pilot, generated earlier' }
    return agent(genPrompt(c), { label: 'gen:' + c.id, phase: 'Generate', schema: GEN_SCHEMA })
  },
  async (gen, c) => {
    if (!gen) return null
    const [label, span] = await parallel([
      () => agent(auditPrompt(c, 'label'), { label: 'audit-label:' + c.id, phase: 'Audit', schema: FIX_SCHEMA }),
      () => agent(auditPrompt(c, 'span'), { label: 'audit-span:' + c.id, phase: 'Audit', schema: FIX_SCHEMA }),
    ])
    log(c.id + ': ' + gen.rows + ' rows; label fixes ' + (label ? label.fixes.length : 'n/a') + ', span fixes ' + (span ? span.fixes.length : 'n/a'))
    return { chunk: c.id, style: c.style || 'mixed', files: filesOf(c), gen, label, span }
  },
)
const missing = chunks.filter((c, i) => !results[i]).map(c => c.id)
if (missing.length) log('chunks without a result: ' + missing.join(', '))
return { results: results.filter(Boolean), missing }
