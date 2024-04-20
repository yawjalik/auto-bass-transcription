export interface Transcription {
  name: string,
  lilypond: string,
  pdf?: string,
}

export interface AudioFile {
  name: string,
  audio: string,
}

export interface AudioSample extends AudioFile {
  lilypond?: string,
}

export interface TranscriptionRequest extends AudioFile {
  model: string
}