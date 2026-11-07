"""DPGuard-style prompting baseline (open VLM re-implementation, NOT the official code).

Uses the final prompt published in Appendix D of Shi et al., "50 Shades of Deceptive
Patterns" (WWW'25, arXiv 2501.13351) with a local open-weight VLM. It omits DPGuard's
ResNet101 pre-filter (needs their training data) and its commercial MLLM. Replaced by
the official release if the authors grant access.

ContextDP-web labels urgency/scarcity types that DPGuard's 21-category taxonomy has no
class for, so only two types map cleanly:
  Preselection   -> DEFAULT CHOICE
  False Hierarchy-> ATTENTION DISTRACTION
and the any-pattern decision uses all 21 categories.
"""
import argparse
import json
import time
from pathlib import Path

from eval.contextdp import DATA_DIR, load_labels, score_image_level
from eval.vlm_baseline import OLLAMA, encode
import urllib.request

CATS = ["Nagging", "Roach Motel", "Price Comparison Prevention", "Intermediate Currency",
        "Forced Continuity", "Hidden Costs", "Sneak into Basket", "Hidden Information",
        "Preselection", "Toying with Emotion", "False Hierarchy", "Disguised Ads",
        "Tricked Questions", "Small Close Button", "Social Pyramid", "Privacy Zuckering",
        "Gamification", "Countdown on Ads", "Watch Ads to Unlock", "Pay to Avoid Ads",
        "Forced Enrollment"]
MAP = {"Preselection": "DEFAULT CHOICE", "False Hierarchy": "ATTENTION DISTRACTION"}

PROMPT = (
    "Kindly conduct a comprehensive analysis of the given image to identify any deceptive "
    "patterns, employing the provided taxonomy. While examining the image, consider the "
    "following categories of deceptive patterns:\n"
    "1. **Nagging**: Spot any repetitive and unexpected pop-up windows that disrupt user activities. "
    "2. **Roach Motel**: Identify scenarios where opting in is easy, but opting out is complicated or obscure. "
    "3. **Price Comparison Prevention**: Detect any elements that hinder direct comparisons of prices or plans. "
    "4. **Intermediate Currency**: Look for instances where virtual currencies are used to obscure real financial costs. "
    "5. **Forced Continuity**: Examine if users are charged after a trial period ends without clear advance notice. "
    "6. **Hidden Costs**: Check for the late disclosure of additional costs such as taxes, delivery, or service fees. "
    "7. **Sneak into Basket**: Identify if items not actively selected by users are automatically added to the shopping cart. "
    "8. **Hidden Information**: Ensure that essential options or actions are not concealed or difficult to find. "
    "9. **Preselection**: Look for cases where options are preselected by default, without explicit user consent. "
    "10. **Toying with Emotion**: Evaluate design elements such as language, colors, or styles that are intended to "
    "elicit emotional responses and pressure users into making decisions. "
    "11. **False Hierarchy**: Observe if any option is made to appear more significant than other equivalent choices. "
    "12. **Disguised Ads**: Detect if advertisements are designed to resemble normal content. "
    "13. **Tricked Questions**: Identify any confusing or misleadingly worded questions. "
    "14. **Small Close Button**: Check if close buttons are too small to be easily located or clicked. "
    "15. **Social Pyramid**: Look for incentives encouraging users to share content with friends for rewards. "
    "16. **Privacy Zuckering**: Assess if default options necessitate sharing unnecessary personal information. "
    "17. **Gamification**: Note if users are required to repeatedly perform tasks to earn rewards. "
    "18. **Countdown on Ads**: Identify if timers restrict users from closing ads immediately. "
    "19. **Watch Ads to Unlock**: Check if users must watch advertisements to access particular features or rewards. "
    "20. **Pay to Avoid Ads**: Note if users are charged to remove adverts. "
    "21. **Forced Enrollment**: Determine if users must sign up or sign in before they can utilize the service.\n"
    "Additional steps for a thorough review: - **UI Color and Text**: Assess the user interface design elements "
    "such as color schemes and text that might manipulate user decisions. Focus particularly on color contrasts and "
    "overall readability. - **Information Visibility**: Verify that critical information is easily accessible without "
    "reliance on hidden links or convoluted navigation paths. - **User Experience Flow**: Evaluate the user journey to "
    "ensure smooth navigation with minimal interruptions. - **Call-to-Action Design**: Examine the visibility and "
    "clarity of call-to-action buttons and links compared to dismissive or less prominent options. - **Consistency**: "
    "Ensure that design elements and user expectations remain consistent across different pages or sections.\n"
    "Your detailed review and precise identification of any deceptive patterns in this image are crucial to understand "
    "their impact and effectiveness.\n"
    'Answer as JSON: {"patterns": [<category names exactly as written above; empty list if none>]}')


def ask(model, image_b64, seed=42):
    body = {"model": model, "stream": False,
            "messages": [{"role": "user", "content": PROMPT, "images": [image_b64]}],
            "format": {"type": "object", "properties": {"patterns": {
                "type": "array", "items": {"type": "string", "enum": CATS}}},
                "required": ["patterns"]},
            "options": {"temperature": 0, "seed": seed, "num_ctx": 8192}}
    req = urllib.request.Request(OLLAMA, json.dumps(body).encode(), {"Content-Type": "application/json"})
    out = json.load(urllib.request.urlopen(req, timeout=600))
    return json.loads(out["message"]["content"]).get("patterns", [])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen2.5vl:7b")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    labels = load_labels()
    names = sorted(labels)
    cache_path = Path(a.out) / "dpguard_style_predictions.json"
    cache = json.load(open(cache_path)) if cache_path.exists() else {}
    t0 = time.time()
    for i, n in enumerate(names):
        if n in cache:
            continue
        cache[n] = ask(a.model, encode(DATA_DIR / "images" / n))
        json.dump(cache, open(cache_path, "w"), indent=0)
        print(f"{i + 1}/{len(names)} {n[:40]:40s} -> {cache[n]} ({time.time() - t0:.0f}s)", flush=True)
    print("done:", len(cache))


if __name__ == "__main__":
    main()
