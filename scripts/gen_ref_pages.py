"""Generate the code reference pages."""

from pathlib import Path

import mkdocs_gen_files

nav = mkdocs_gen_files.Nav()
mod_symbol = '<code class="doc-symbol doc-symbol-nav doc-symbol-module"></code>' # nuevo

src = Path(__file__).parent.parent / "src"

for path in sorted(src.rglob("*.py")):  
    module_path = path.relative_to(src).with_suffix("")  
    doc_path = path.relative_to(src / "rapidae").with_suffix(".md")  
    full_doc_path = Path("reference", doc_path)

    parts = tuple(module_path.parts) #list(module_path.parts) # nuevo

    if parts[-1] == "__init__":  
        parts = parts[:-1]
        doc_path = doc_path.with_name("index.md")
        full_doc_path = full_doc_path.with_name("index.md")
    elif parts[-1].startswith("_"): #parts[-1] == "__main__": # nuevo
        continue
    
    #nav[parts] = doc_path.as_posix()
    nav_parts = [f"{mod_symbol} {part}" for part in parts] # nuevo
    nav[tuple(nav_parts)] = doc_path.as_posix()

    with mkdocs_gen_files.open(full_doc_path, "w") as fd:  
        ident = ".".join(parts)  
        #print("::: " + identifier, file=fd)
        fd.write(f"::: {ident}") # nuevo

    mkdocs_gen_files.set_edit_path(full_doc_path, ".." / path)  

with mkdocs_gen_files.open("reference/SUMMARY.md", "w") as nav_file:  
    nav_file.writelines(nav.build_literate_nav())  



