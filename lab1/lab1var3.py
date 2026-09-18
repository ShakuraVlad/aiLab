TRANSITIONS = {
    ('q0', 'a'): 'q0',
    ('q0', 'b'): 'q1',
    ('q1', 'd'): 'q2',
    ('q2', 'd'): 'q2',
}

START_STATE = 'q0'
FINAL_STATES = {'q2'}


def is_accepted(word):
    state = START_STATE
    for ch in word:
        state = TRANSITIONS.get((state, ch), 'err')
        if state == 'err':
            return False
    return state in FINAL_STATES


def main():
    tests = [
        ('bd', True),
        ('abd', True),
        ('aabd', True),
        ('bdd', True),
        ('aaabddd', True),
        ('', False),
        ('a', False),
        ('b', False),
        ('ab', False),
        ('bdb', False),
        ('aabdc', False),
        ('dabd', False),
        ('aaabbd', False),
    ]

    print("Проверка автомата на тестовых словах:")
    for word, expected in tests:
        got = is_accepted(word)
        mark = "OK" if got == expected else "ОШИБКА"
        print(f"  '{word}' -> {got} (ожидалось {expected}) [{mark}]")

    print()
    word = input("Введите слово для проверки: ")
    if is_accepted(word):
        print("Слово ДОПУСТИМО")
    else:
        print("Слово НЕДОПУСТИМО")


if __name__ == '__main__':
    main()
