/****************************************************************************
**
** Copyright (C) 2009-2010 TECHNOGERMA Systems France and/or its subsidiary(-ies).
** Contact: Technogerma Systems France Information (contact@technogerma.fr)
**
** This file is part of the CAMP library.
**
** CAMP is free software: you can redistribute it and/or modify
** it under the terms of the GNU Lesser General Public License as published by
** the Free Software Foundation, either version 3 of the License, or
** (at your option) any later version.
** 
** CAMP is distributed in the hope that it will be useful,
** but WITHOUT ANY WARRANTY; without even the implied warranty of
** MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
** GNU Lesser General Public License for more details.
** 
** You should have received a copy of the GNU Lesser General Public License
** along with CAMP.  If not, see <http://www.gnu.org/licenses/>.
**
****************************************************************************/


#ifndef CAMP_ENUMOBJECT_HPP
#define CAMP_ENUMOBJECT_HPP


#include <camp/config.hpp>
#include <camp/enumget.hpp>
#include <boost/type_traits.hpp>
#include <boost/operators.hpp>
#include <boost/utility/enable_if.hpp>
#include <string>


namespace camp
{
/**
 * \brief Wrapper to manipulate enumerated values in the CAMP system
 *
 * camp::EnumObject is an abstract representation of enum values, and supports
 * conversions from strings and integers.
 *
 * \sa UserObject
 */
class CAMP_API EnumObject : boost::totally_ordered<EnumObject>
{
public:

    /**
     * \brief Construct the enum object from an enumerated value
     *
     * \param value Value to store in the enum object
     */
    template <typename T>
    EnumObject(T value, typename boost::enable_if<boost::is_enum<T> >::type* = 0);

    /**
     * \brief Get the value of the enum object
     *
     * \return Integer value of the enum object
     */
    long value() const;

    /**
     * \brief Get the name of the enum object
     *
     * \return String containing the name of the enum object
     */
    const std::string& name() const;

    /**
     * \brief Retrieve the metaenum of the stored enum object
     *
     * \return Reference to the object's metaenum
     */
    const Enum& getEnum() const;

    /**
     * \brief Operator == to compare equality between two enum objects
     *
     * Two enum objects are equal if their metaenums and values are both equal
     *
     * \param other Enum object to compare with this
     *
     * \return True if both enum objects are the same, false otherwise
     */
    bool operator==(const EnumObject& other) const;

    /**
     * \brief Operator < to compare two enum objects
     *
     * \param other Enum object to compare with this
     *
     * \return True if this < other
     */
    bool operator<(const EnumObject& other) const;

private:

    long m_value; ///< Value
    const Enum* m_enum; ///< Metaenum associated to the value
};

} // namespace camp

#include <camp/enumobject.inl>


#endif // CAMP_ENUMOBJECT_HPP
