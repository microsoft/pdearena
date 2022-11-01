# This script generates the research page from the research.yml file.

import yaml

header = """
# Research

Following is a list of research papers that have been published using the PDEArena framework.

If you have used PDEArena in your research, and would like it listed here, please add your paper to [this file](https://github.com/microsoft/pdearena/blob/main/docs/research.yml) by sending a pull request to the [PDEArena repository](https://github.com/microsoft/pdearena).
"""

def snippet(paper):
    title = paper["title"]
    authors = (
        ", ".join(paper["authors"]).replace("(", "<sup>").replace(")", "</sup>")
    )
    affiliations = ", ".join(
        f"<sup>{num}</sup>{affil}" for num, affil in paper["affiliations"].items()
    )
    link = paper["link"]
    abstract = paper["abstract"]
    image_file = paper["image"]

    paper_snippet = f"""

<figure markdown>
![](assets/images/research/{image_file}){{ width="500"}}
<figcaption>
<!-- Large font: -->
<h2>
<a href="{link}">{title}</a>
</h2>
</figcaption>
</figure>
<center>
{authors}
    
<small>{affiliations}</small>
</center>
**Abstract:** {abstract}\n\n

    """
    return paper_snippet


def main(outfile):
    with open("research.yml", "r") as f:
        research = yaml.load(f, Loader=yaml.SafeLoader)["papers"]

    with open(outfile, "w") as f:
        f.write(header)    

        research = sorted(research, key=lambda x: x["date"], reverse=True)

        snippets = [snippet(paper) for paper in research]

        f.write("\n\n---\n\n".join(snippets))

if __name__ == "__main__":
    main("research.md")
            

