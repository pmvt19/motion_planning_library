import time

class Timer:
    def __init__(self):
        self.times: list[tuple[str, float]] = []

    def mark_time(self, breakpoint_name: str):
        self.times.append((breakpoint_name, time.time()))

    def __str__(self):
        results: dict[str, float] = {}
        for i in range(len(self.times) - 1):
            breakpoint_name, timestamp1 = self.times[i]
            _, timestamp2 = self.times[i+1]
            results[breakpoint_name] = timestamp2 - timestamp1

        formatted_results = ""
        for result in results:
            formatted_results += f"{result}: {results[result]}\n"
        return formatted_results