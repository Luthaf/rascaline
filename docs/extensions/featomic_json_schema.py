import copy
import json
import os

from docutils import nodes
from docutils.parsers.rst import Directive
from html_hidden import html_hidden
from markdown_it import MarkdownIt
from myst_parser.config.main import MdParserConfig
from myst_parser.mdit_to_docutils.base import DocutilsRenderer


def markdown_to_docutils(text):
    parser = MarkdownIt()
    tokens = parser.parse(text)

    renderer = DocutilsRenderer(parser)
    return renderer.render(tokens, {"myst_config": MdParserConfig()}, {})


def _target_id(text):
    return nodes.make_id("json-schema-" + text)


class JsonSchemaDirective(Directive):
    required_arguments = 1

    def __init__(self, *args, **kwargs):
        super(JsonSchemaDirective, self).__init__(*args, **kwargs)
        self.docs_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        # use Dict[str, None] as an ordered set
        self._definitions = {}

    def run(self, *args, **kwargs):
        schema, content = self._load_schema()

        title = f"{schema['title']} hyper-parameters"
        root_target, section = self._transform(schema, title)

        schema_node = html_hidden(toggle="Show full JSON schema")
        schema_node += nodes.literal_block(text=content)
        section.insert(1, schema_node)

        # add missing entries to  self._definitions
        for name in schema.get("$defs", {}).keys():
            self._definition_used(name)

        for name in self._definitions.keys():
            definition = schema["$defs"][name]
            target, subsection = self._transform(definition, name)

            section += target
            section += subsection

        return [root_target, section]

    def _definition_used(self, name):
        self._definitions[name] = None

    def _transform(self, schema, name):
        target_id = _target_id(name)
        target = nodes.target(
            "", "", ids=[target_id], names=[target_id], line=self.lineno
        )

        section = nodes.section()
        section += nodes.title(text=name)

        description = schema.get("description", "")
        section.extend(markdown_to_docutils(description))

        section += self._json_schema_to_nodes(schema)

        return (target, section)

    def _load_schema(self):
        path = os.path.join(self.docs_root, self.arguments[0])
        if not os.path.exists(path):
            raise Exception(f"Unable to find JSON schema at '{path}'.")

        self.state.document.settings.env.note_dependency(path)

        with open(path) as fd:
            content = fd.read()

        schema = json.loads(content)

        schema["$$rust-type"] = schema["title"]
        schema["title"] = os.path.basename(path).split(".")[0]

        return schema, content

    def _json_schema_to_nodes(
        self,
        schema,
        inline=False,
        description=True,
        optional=False,
    ):
        """Transform the schema for a single type to docutils nodes"""

        if optional:
            # can only use optional for inline mode
            assert inline

        optional_str = "?" if optional else ""

        if "$ref" in schema:
            assert "properties" not in schema
            assert "oneOf" not in schema
            assert "anyOf" not in schema
            assert "allOf" not in schema

            ref = schema["$ref"]
            assert ref.startswith("#/$defs/")
            type_name = ref.split("/")[-1]

            self._definition_used(type_name)

            refid = _target_id(type_name)
            container = nodes.generated()
            container += nodes.reference(
                internal=True,
                refid=refid,
                text=type_name + optional_str,
            )

            return container

        # enums values are represented as allOf
        if "allOf" in schema:
            assert "properties" not in schema
            assert "oneOf" not in schema
            assert "anyOf" not in schema
            assert "$ref" not in schema

            assert len(schema["allOf"]) == 1
            return self._json_schema_to_nodes(schema["allOf"][0])

        # Enum variants uses "oneOf"
        if "oneOf" in schema:
            assert "anyOf" not in schema
            assert "allOf" not in schema
            assert "$ref" not in schema

            container = nodes.paragraph()
            container += nodes.Text(
                'Pick one of the following according to its "type":'
            )

            global_properties = copy.deepcopy(schema.get("properties", {}))

            for prop in global_properties.values():
                prop["description"] = "See below."

            bullet_list = nodes.bullet_list()
            for possibility in schema["oneOf"]:
                possibility = copy.deepcopy(possibility)
                possibility["properties"].update(global_properties)

                item = nodes.list_item()
                item += self._json_schema_to_nodes(
                    possibility, inline=True, description=False
                )

                description = possibility.get("description", "")
                item.extend(markdown_to_docutils(description))

                item += self._json_schema_to_nodes(possibility, inline=False)

                bullet_list += item

            container += bullet_list

            global_properties = copy.deepcopy(schema)
            global_properties.pop("oneOf")
            if "properties" in global_properties:
                container += nodes.transition()
                container += self._json_schema_to_nodes(global_properties, inline=False)

            return container

        if "anyOf" in schema:
            assert "properties" not in schema
            assert "oneOf" not in schema
            assert "allOf" not in schema
            assert "$ref" not in schema

            # only supported for Option<T>
            assert len(schema["anyOf"]) == 2
            assert schema["anyOf"][1]["type"] == "null"
            return self._json_schema_to_nodes(
                schema["anyOf"][0], inline=True, optional=optional
            )

        if "type" in schema:
            assert "oneOf" not in schema
            assert "anyOf" not in schema
            assert "allOf" not in schema
            assert "$ref" not in schema

            if schema["type"] == "null":
                assert not optional
                return nodes.literal(text="null")

            elif schema["type"] == "object":
                assert not optional
                if not inline:
                    field_list = nodes.field_list()
                    for name, content in schema.get("properties", {}).items():
                        name = nodes.field_name(text=name)
                        name += nodes.Text(": ")
                        if "default" in content:
                            name += nodes.Text("optional, ")

                        name += self._json_schema_to_nodes(
                            content, inline=True, optional=False
                        )

                        field_list += name

                        if description:
                            description_text = content.get("description", "")

                            description = markdown_to_docutils(description_text)
                            body = nodes.field_body()
                            body.extend(description)

                            field_list += body

                    additional = schema.get("additionalProperties")
                    if additional is not None:
                        pass

                    return field_list
                else:
                    object_node = nodes.inline()

                    object_node += nodes.Text("{")

                    fields_unordered = schema.get("properties", {})
                    # put "type" first in the output
                    fields = {}
                    if "type" in fields_unordered:
                        fields["type"] = fields_unordered.pop("type")
                    fields.update(fields_unordered)

                    n_fields = len(fields)
                    for i_field, (name, content) in enumerate(fields.items()):
                        field = nodes.inline()
                        field += nodes.Text(f'"{name}": ')

                        subfields = self._json_schema_to_nodes(
                            content,
                            inline=True,
                            optional="default" in content,
                        )
                        if isinstance(subfields, nodes.literal):
                            subfields = [subfields]

                        field += subfields

                        if i_field != n_fields - 1:
                            field += nodes.Text(", ")

                        object_node += field

                    additional = schema.get("additionalProperties")
                    if isinstance(additional, dict):
                        # JSON Schema does not have a concept of key type being anything
                        # else than string. In featomic, we annotate `HashMap` with a
                        # custom `x-key-type` to carry this information all the way to
                        # here
                        key_type = schema.get("x-key-type")
                        if key_type is None:
                            key_type = "string"

                        field = nodes.inline()
                        field += nodes.Text("[key: ")
                        field += nodes.literal(text=key_type)
                        field += nodes.Text("]: ")

                        field += self._json_schema_to_nodes(additional)

                        object_node += field

                    object_node += nodes.Text("}")

                    return object_node

            elif schema["type"] == "number":
                assert schema["format"] == "double"
                return nodes.literal(text="number" + optional_str)

            elif schema["type"] == "integer":
                if "format" not in schema:
                    return nodes.literal(text="integer" + optional_str)

                if schema["format"].startswith("int"):
                    return nodes.literal(text="integer" + optional_str)
                elif schema["format"].startswith("uint"):
                    return nodes.literal(text="positive integer" + optional_str)
                else:
                    raise Exception(f"unknown integer format: {schema['format']}")

            elif schema["type"] == "string":
                assert not optional
                if "enum" in schema:
                    values = [f'"{v}"' for v in schema["enum"]]
                    return nodes.literal(text=" | ".join(values))
                elif "const" in schema:
                    return nodes.Text('"' + schema["const"] + '"')
                else:
                    return nodes.literal(text="string")

            elif schema["type"] == "boolean":
                if optional:
                    return nodes.literal(text="boolean?")
                else:
                    return nodes.literal(text="boolean")

            elif isinstance(schema["type"], list):
                # we only support list for Option<T>
                assert len(schema["type"]) == 2
                assert schema["type"][1] == "null"

                schema["type"] = schema["type"][0]
                return self._json_schema_to_nodes(
                    schema, inline=True, optional=optional
                )

            elif schema["type"] == "array":
                assert not optional
                array_node = nodes.inline()
                inner = self._json_schema_to_nodes(schema["items"], inline=True)
                if isinstance(inner, nodes.literal):
                    array_node += nodes.literal(text=inner.astext() + "[]")
                else:
                    array_node += inner
                    array_node += nodes.Text("[]")
                return array_node

            else:
                raise Exception(f"unsupported JSON type ({schema['type']}) in schema")

        raise Exception(f"unsupported JSON schema: {schema}")


def setup(app):
    app.require_sphinx("3.3")
    app.add_directive("featomic-json-schema", JsonSchemaDirective)
