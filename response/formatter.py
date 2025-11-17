class ResponseFormatter:
    @staticmethod
    def format(ai_text: str):
        if not ai_text:
            ai_text = ""

        # Remove double newlines → single newline
        ai_text = ai_text.replace("\n\n", "\n")

        # Remove excessive whitespace
        ai_text = "\n".join([line.strip() for line in ai_text.split("\n")])

        # Collapse multiple blank lines
        lines = [line for line in ai_text.split("\n") if line.strip() != ""]
        ai_text = "\n".join(lines)

        # Trim start & end
        ai_text = ai_text.strip()

        return {"ai_text": ai_text}
