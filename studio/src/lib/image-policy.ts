/** The image-mode policy, beside `audioPolicy` and shaped like it: one rule
 *  for the web composer and any native presentation of it.
 *
 *  The composer is in IMAGE mode when every lane makes pictures and at least
 *  one cannot chat - then the text box is the prompt and the reply is a
 *  picture, and there is no sampling to set. A lane set where everything can
 *  also chat stays a chat (a future model that both talks and draws would
 *  draw on request, not by default). Mixed sets are refused upstream by the
 *  shared-input test; here they simply read as not-image. */
export interface ImageLane {
  chat: boolean
  image: boolean
}

export function imagePolicy(lanes: ImageLane[]) {
  const imageOk = lanes.length > 0 && lanes.every((l) => l.image)
  const imageMode = imageOk && !lanes.every((l) => l.chat)
  return { imageOk, imageMode }
}
