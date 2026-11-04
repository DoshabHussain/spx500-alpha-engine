import sys

with open('src/web/app.py', 'r', encoding='utf-8') as f:
    text = f.read()

replacements = {
    '\u00b7': '|',
    '\U0001f680': '[GO]',
    '\U0001f449': '->',
}

for k, v in replacements.items():
    text = text.replace(k, v)

rem = {c: hex(ord(c)) for c in text if ord(c) > 127}
print(f'Remaining non-ascii count: {len(rem)}')
for c, h in rem.items():
    print(f'  char code: {h}')

with open('src/web/app.py', 'w', encoding='utf-8') as f:
    f.write(text)

print('Updated src/web/app.py')
