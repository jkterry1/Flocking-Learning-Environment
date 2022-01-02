pip install -e .
# stolen from https://stackoverflow.com/questions/525592/find-and-replace-inside-a-text-file-from-a-bash-command
# removes the magent import so tests can run properly (there is no magent)
sed -i -e 's/import magent/#/g' fle/flocking_env.py
