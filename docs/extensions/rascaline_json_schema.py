import os
import json

from docutils.parsers.rst import Directive
from docutils import nodes

from recommonmark.parser import CommonMarkParser


def _target_id(text):
    return nodes.make_id("json-schema-" + text)


class html_hidden(nodes.Element):
    """
    A node that will be hidden by default in HTML output
    """

    pass


def visit_html_hidden_latex(self, node):
    pass


def depart_html_hidden_latex(self, node):
    pass


def visit_html_hidden_html(self, node):
    self.body.append("<details><summary><a>{}</a></summary>".format(node["toggle"]))


def depart_html_hidden_html(self, node):
    self.body.append("</details>")


class JsonSchemaDirective(Directive):
    required_arguments = 1

    def __init__(self, *args, **kwargs):
        super(JsonSchemaDirective, self).__init__(*args, **kwargs)
        self._inline_call_count = 0

    def run(self, *args, **kwargs):
        schema, content = self._load_schema()

        title = f"{schema['title']} hyper-parameters"
        root_target, section = self._transform(schema, title)

        schema_node = html_hidden(toggle="Show full JSON schema")
        schema_node += nodes.literal_block(text=content)
        section.insert(1, schema_node)

        for name, definition in schema.get("definitions", {}).items():
            target, subsection = self._transform(definition, name)

            section += target
            section += subsection

        return [root_target, section]

    def _transform(self, schema, name):
        target_id = _target_id(name)
        target = nodes.target(
            "", "", ids=[target_id], names=[target_id], line=self.lineno
        )

        section = nodes.section()
        section += nodes.title(text=name)

        self._parse_md_to(section, schema.get("description", ""))

        section += self._json_schema_to_nodes(schema)

        return (target, section)

    def _load_schema(self):
        path = self.arguments[0]
        if not os.path.exists(path):
            raise Exception(f"unable to find JSON schema at '{path}'")

        self.state.document.settings.env.note_dependency(os.path.abspath(path))

        with open(path) as fd:
            content = fd.read()

        schema = json.loads(content)

        schema["$$rust-type"] = schema["title"]
        schema["title"] = os.path.basename(path).split(".")[0]

        return schema, content

    def _json_schema_to_nodes(self, schema, inline=False):
        """Transform the schema for a single type to docutils nodes"""

        if "type" in schema:
            if schema["type"] == "object":
                if not inline:
                    field_list = nodes.field_list()
                    for name, content in schema.get("properties", {}).items():
                        name = nodes.field_name(text=name)
                        name += nodes.Text(": ")
                        name += self._json_schema_to_nodes(content, inline=True)

                        field_list += name
                        body = nodes.field_body()

                        self._parse_md_to(body, content.get("description", ""))

                        field_list += body

                    return field_list
                else:
                    self._inline_call_count += 1

                    object_node = nodes.inline()

                    if self._inline_call_count > 1:
                        object_node += nodes.Text("{")

                    for name, content in schema.get("properties", {}).items():
                        field = nodes.inline()
                        field += nodes.Text(f"{name}: ")

                        subfields = self._json_schema_to_nodes(content, inline=True)
                        for (i, sf) in enumerate(subfields):
                            field += sf

                            if isinstance(sf, nodes.inline):
                                if i != len(subfields) - 2:
                                    # len(xxx) - 2 to account for the final }
                                    field += nodes.Text(", ")

                        object_node += field

                    if self._inline_call_count > 1:
                        object_node += nodes.Text("}")

                    self._inline_call_count -= 1
                    return object_node

            elif schema["type"] == "number":
                assert schema["format"] == "double"
                return nodes.literal(text="number")

            elif schema["type"] == "integer":
                if schema["format"] == "int":
                    return nodes.literal(text="signed integer")
                elif schema["format"] == "uint":
                    return nodes.literal(text="unsigned integer")
                else:
                    raise Exception("unknown integer format")

            elif schema["type"] == "string":
                # TODO enums?
                return nodes.literal(text="string")

            elif schema["type"] == "boolean":
                return nodes.literal(text="boolean")

            else:
                raise Exception(f"unsupported JSON type ({schema['type']}) in schema")

        # enums values are represented as allOf
        if "allOf" in schema:
            assert len(schema["allOf"]) == 1
            ref = schema["allOf"][0]["$ref"]
            assert ref.startswith("#/definitions/")
            type_name = ref.split("/")[-1]

            refid = _target_id(type_name)

            return nodes.reference(internal=True, refid=refid, text=type_name)

        # Enum variants uses "oneOf"
        if "oneOf" in schema:
            bullet_list = nodes.bullet_list()
            for possibility in schema["oneOf"]:
                item = nodes.list_item()
                item += self._json_schema_to_nodes(possibility, inline=True)
                self._parse_md_to(item, possibility.get("description", ""))
                bullet_list += item

            return bullet_list

        raise Exception(f"unsupported JSON schema: {schema}")

    def _parse_md_to(self, node, content):
        # HACK: make the CommonMarkParser think that `node` is actually the full
        # document
        assert not hasattr(node, "reporter")
        assert not hasattr(node, "note_parse_message")

        node.reporter = self.state.document.reporter
        node.note_parse_message = self.state.document.note_parse_message

        md_parser = CommonMarkParser()
        md_parser.parse(content, node)

        del node.reporter
        del node.note_parse_message


def setup(app):
    app.require_sphinx("3.3")
    app.add_directive("rascaline-json-schema", JsonSchemaDirective)

    app.add_node(
        html_hidden,
        html=(visit_html_hidden_html, depart_html_hidden_html),
        latex=(visit_html_hidden_latex, depart_html_hidden_latex),
    )
