import os

class VideoWriter:
    def __init__(self, output_file="vehicle_counts.txt"):
        self.output_file = output_file

        if os.path.exists(self.output_file):
            os.remove(self.output_file)

    def write_counts(self, counts):
        """
        Escribe los contadores en el archivo en formato:
        NORD: X  SUD: Y  TOTAL: Z
        """
        total = counts.get('north', 0) + counts.get('south', 0)
        line = f"NORD: {counts.get('north', 0)}  SUD: {counts.get('south', 0)}  TOTAL: {total}\n"
        with open(self.output_file, "a") as f:
            f.write(line)

    # def write_final_counts(self, counts, total_detected=0):
    #     """Escriu un resum final dels comptatges i cotxes detectats"""
    #     total = counts.get('north', 0) + counts.get('south', 0)
    #     line = (
    #         f"Resumen final:\n"
    #         f"NORD: {counts.get('north', 0)}\n"
    #         f"SUD: {counts.get('south', 0)}\n"
    #         f"TOTAL: {total}\n"
    #         f"COTXES DETECTATS: {total_detected}\n"
    #     )
    #     with open(self.output_file, "a") as f:
    #         f.write(line)